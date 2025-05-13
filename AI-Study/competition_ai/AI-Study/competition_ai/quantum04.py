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
try:
    from hybrid_extractor import HybridExtractor
except ImportError:
    class HybridExtractor:
        def __init__(self, *args, **kwargs):
            print("Warning: hybrid_extractor not installed. Using stub.")
        def extract(self, *args, **kwargs):
            raise NotImplementedError
import pennylane as qml
from tqdm import tqdm
from datetime import datetime

# ── Set seed for reproducibility ─────────────────
SEED = 44
random.seed(SEED) 
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Config ───────────────────────────────────────
BATCH_SIZE = 512
LR = 5e-4
EPOCHS = 200
PATIENCE = 10
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Augmentation & Split ─────────────────────
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomErasing(p=0.5),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
full_train = torchvision.datasets.FashionMNIST(
    "./", train=True, download=True, transform=transform_train)
test_ds = torchvision.datasets.FashionMNIST(
    "./", train=False, download=True, transform=transform_test)
# filter 0 vs 6, map 6->1
mask = (full_train.targets==0)|(full_train.targets==6)
full_train.targets[full_train.targets==6]=1
binary_ds = Subset(full_train, torch.where(mask)[0])
# train/val split 90/10
t_val=int(0.1*len(binary_ds))
train_ds,val_ds=random_split(binary_ds,[len(binary_ds)-t_val,t_val])
train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True)
val_loader  =DataLoader(val_ds,  BATCH_SIZE,shuffle=False)

# ── MixUp ────────────────────────────────────────
def mixup_data(x,y,alpha=1.0):
    if alpha>0:
        lam=np.random.beta(alpha,alpha)
    else:
        lam=1
    idx=torch.randperm(x.size(0)).to(device)
    mixed=lam*x+(1-lam)*x[idx]
    return mixed, y, y[idx], lam

# ── Quantum Circuit ──────────────────────────────
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev=qml.device("default.qubit",wires=2)
        self.params=Parameter(torch.randn(12,dtype=torch.float64),requires_grad=True)
        self.obs=qml.PauliZ(0)@qml.PauliZ(1)
        @qml.qnode(self.dev,interface="torch",diff_method="backprop")
        def circuit(x):
            qml.AngleEmbedding(x,wires=[0,1])
            for _ in range(2):
                qml.StronglyEntanglingLayers(self.params.reshape(2,2,3),wires=[0,1])
            return qml.expval(self.obs)
        self.circuit=circuit
    def forward(self,x):
        return self.circuit(x)

# ── Hybrid Model ─────────────────────────────────
class HybridCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(1,8,3,padding=1);self.bn1=BatchNorm2d(8)
        self.conv2=Conv2d(8,16,3,padding=1);self.bn2=BatchNorm2d(16)
        self.conv3=Conv2d(16,32,3,padding=1);self.bn3=BatchNorm2d(32)
        self.dropout=Dropout2d(0.4)
        self.fc1=Linear(32*3*3,64)
        self.fc2=Linear(64,2)
        self.qnn=QuantumCircuit()
        self.final=Linear(1,2)
    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)));x=F.max_pool2d(x,2)
        x=F.relu(self.bn2(self.conv2(x)));x=F.max_pool2d(x,2)
        x=F.relu(self.bn3(self.conv3(x)));x=F.max_pool2d(x,2)
        x=self.dropout(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x));x=self.fc2(x)
        q_out=torch.stack([self.qnn(vec) for vec in x]).view(-1,1).float()
        return self.final(q_out)

model=HybridCNN().to(device)
# param check
total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
assert total_params<=50000

# optimizer+scheduler
optimizer=Adam(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=OneCycleLR(optimizer,max_lr=LR*10,epochs=EPOCHS,steps_per_epoch=len(train_loader))
loss_fn=CrossEntropyLoss(label_smoothing=0.1)
best_val=1e9;pat=0;best_path=None

for ep in range(1,EPOCHS+1):
    model.train();tl=0
    for data,target in tqdm(train_loader,desc=f"Train {ep}"):
        data,target=data.to(device),target.to(device)
        data,ta,tb,lam=mixup_data(data,target)
        optimizer.zero_grad();logits=model(data)
        loss=lam*loss_fn(logits,ta)+(1-lam)*loss_fn(logits,tb)
        loss.backward();clip_grad_norm_(model.parameters(),1.0)
        optimizer.step();scheduler.step()
        tl+=loss.item()
    tl/=len(train_loader)
    model.eval();vl=0;corr=0;tot=0
    with torch.no_grad():
        for d,t in val_loader:
            d,t=d.to(device),t.to(device)
            lg=model(d);vl+=loss_fn(lg,t).item()
            pr=lg.argmax(1);corr+=(pr==t).sum().item();tot+=t.size(0)
    vl/=len(val_loader);acc=corr/tot
    print(f"Ep{ep} TrainLoss{tl:.4f} ValLoss{vl:.4f} ValAcc{acc:.4f}")
    if vl<best_val:
        best_val=vl;pat=0
        path=f"best_model_{datetime.now():%Y%m%d_%H%M%S}.pt"
        torch.save(model.state_dict(),path);best_path=path
        print(f"Saved {path}")
    else:
        pat+=1
        if pat>=PATIENCE:print("EarlyStopping");break

# inference
test_full=torchvision.datasets.FashionMNIST("./",train=False,download=True,transform=transform_test)
lf=DataLoader(test_full,BATCH_SIZE,shuffle=False)
model.load_state_dict(torch.load(best_path));model.eval();res=[]
with torch.no_grad():
    for d,_ in tqdm(lf,desc="Test"):
        pr=model(d.to(device)).argmax(1).cpu().numpy()
        res.extend(np.where(pr==1,6,0).tolist())
assert len(res)==len(test_full)
fn=f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
np.savetxt(fn,res,fmt='%d');print(f"Saved {fn}")


# Ep90 TrainLoss0.5181 ValLoss0.4140 ValAcc0.8567
# Saved best_model_20250730_170053.pt