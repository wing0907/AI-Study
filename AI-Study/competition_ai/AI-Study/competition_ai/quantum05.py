import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss, Module, Conv2d, BatchNorm2d, Dropout2d, Linear
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
# hybrid_extractor stub
# try:
#     from hybrid_extractor import HybridExtractor
# except ImportError:
class HybridExtractor:
    def __init__(self, *args, **kwargs): print("Warning: hybrid_extractor stub.")
    def extract(self, *args, **kwargs): raise NotImplementedError
import pennylane as qml
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ── Seed for reproducibility ─────────────────────
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Config ───────────────────────────────────────
BATCH_SIZE = 256
LR = 5e-4
EPOCHS = 200
PATIENCE = 20
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Augmentation & Split ─────────────────────
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ColorJitter(0.2,0.2,0.2,0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(p=0.5),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
full_train = torchvision.datasets.FashionMNIST("./",train=True,download=True,transform=transform_train)
test_ds    = torchvision.datasets.FashionMNIST("./",train=False,download=True,transform=transform_test)
# filter labels 0 vs 6, map 6->1
mask = (full_train.targets==0)|(full_train.targets==6)
full_train.targets[full_train.targets==6]=1
binary_ds = Subset(full_train, torch.where(mask)[0])
# train/val split 90/10
t_val=int(0.1*len(binary_ds))
train_ds,val_ds=random_split(binary_ds,[len(binary_ds)-t_val,t_val])
train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True)
val_loader  =DataLoader(val_ds,  BATCH_SIZE,shuffle=False)

# ── MixUp & CutMix ───────────────────────────────
def mixup_data(x,y,alpha=0.4):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1
    idx = torch.randperm(x.size(0)).to(device)
    mixed = lam*x + (1-lam)*x[idx]
    return mixed, y, y[idx], lam

# ── Quantum Circuit ──────────────────────────────
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=2)
        self.params = Parameter(torch.randn(18, dtype=torch.float64), requires_grad=True)
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)
        @qml.qnode(self.dev, interface="numpy", diff_method="backprop")
        def circuit(x):
            qml.AngleEmbedding(x, wires=[0,1])
            for _ in range(3):
                qml.StronglyEntanglingLayers(self.params.reshape(3,2,3), wires=[0,1])
            return qml.expval(self.obs)
        self.circuit = circuit
    def forward(self, x):
        return self.circuit(x)

# ── Hybrid Model (channels 12,24,48; fc1=72) ─────────
class HybridCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1,12,3,padding=1); self.bn1 = BatchNorm2d(12)
        self.conv2 = Conv2d(12,24,3,padding=1); self.bn2 = BatchNorm2d(24)
        self.conv3 = Conv2d(24,48,3,padding=1); self.bn3 = BatchNorm2d(48)
        self.dropout= Dropout2d(0.4)
        self.fc1    = Linear(48*3*3,72)
        self.fc2    = Linear(72,2)
        self.qnn    = QuantumCircuit()
        self.final  = Linear(1,2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(x,2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.fc2(x)
                        # Apply QNN on CPU to avoid unsupported CUDA ops
        q_out_list = []
        for v in x:
            # Move to CPU and ensure double precision
            v_cpu = v.cpu().double()
            # Convert to NumPy for QNode
            v_np = v_cpu.detach().numpy()
            # Run quantum circuit on CPU via QuantumCircuit module
            q_val_np = self.qnn(v_np)
            # Wrap NumPy output back to torch Tensor on correct device
            q_val = torch.tensor(q_val_np, device=device, dtype=torch.float32)
            q_out_list.append(q_val)
        # Stack quantum outputs
        q_out = torch.stack(q_out_list).view(-1, 1)
        return self.final(q_out)

model = HybridCNN().to(device)
# parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
assert total_params <= 50000

# optimizer & scheduler
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=LR*10, epochs=EPOCHS, steps_per_epoch=len(train_loader))
loss_fn = CrossEntropyLoss(label_smoothing=0.1)
best_val=1e9; pat=0; best_path=None

for ep in range(1, EPOCHS+1):
    model.train(); tl=0
    for data,target in tqdm(train_loader, desc=f"Train {ep}"):
        data,target = data.to(device), target.to(device)
        data, ta, tb, lam = mixup_data(data, target)
        optimizer.zero_grad(); logits = model(data)
        loss = lam*loss_fn(logits, ta) + (1-lam)*loss_fn(logits, tb)
        loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        tl += loss.item()
    tl /= len(train_loader)
    model.eval(); vl=0; corr=0; tot=0
    with torch.no_grad():
        for d,t in val_loader:
            d,t = d.to(device), t.to(device)
            lg = model(d); vl += loss_fn(lg, t).item()
            pr = lg.argmax(dim=1); corr += (pr==t).sum().item(); tot += t.size(0)
    vl /= len(val_loader); acc = corr/tot
    print(f"Ep{ep} TrainLoss{tl:.4f} ValLoss{vl:.4f} ValAcc{acc:.4f}")
    if vl < best_val:
        best_val=vl; pat=0
        pth=f"best_model_{datetime.now():%Y%m%d_%H%M%S}.pt"
        torch.save(model.state_dict(), pth); best_path=pth
        print(f"Saved {pth}")
    else:
        pat +=1
        if pat>=PATIENCE: print("EarlyStopping"); break

# inference
test_full = torchvision.datasets.FashionMNIST("./", train=False, download=True, transform=transform_test)
loader_full = DataLoader(test_full, BATCH_SIZE, shuffle=False)
model.load_state_dict(torch.load(best_path)); model.eval()
res=[]
with torch.no_grad():
    for d,_ in tqdm(loader_full, desc="Test"):
        pr = model(d.to(device)).argmax(dim=1).cpu().numpy()
        res.extend(np.where(pr==1,6,0).tolist())
assert len(res)==len(test_full)
fn=f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
np.savetxt(fn, res, fmt='%d'); print(f"Saved {fn}")
