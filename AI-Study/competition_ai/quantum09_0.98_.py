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
import pennylane as qml
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score

# ── 1) Reproducibility ─────────────────────────────
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 2) Config ───────────────────────────────────────
BATCH_SIZE      = 256
MAX_LR          = 1e-2   # OneCycleLR 상한
LR              = 5e-4   # ← define it here
EPOCHS          = 50     # 빠른 실험
PATIENCE        = 10
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR  = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ── 3) MixUp ────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# ── 4) Data Preparation ─────────────────────────────
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

full = torchvision.datasets.FashionMNIST(".", train=True, download=True, transform=transform_train)
# 0 vs 6 only
mask = (full.targets == 0) | (full.targets == 6)
full.targets[full.targets == 6] = 1
binary = Subset(full, torch.where(mask)[0])
val_sz = int(0.1 * len(binary))
train_ds, val_ds = random_split(binary, [len(binary)-val_sz, val_sz])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_full    = torchvision.datasets.FashionMNIST(".", train=False, download=True, transform=transform_test)

# ── 5) QuantumCircuit ───────────────────────────────
try:
    dev = qml.device("lightning.qubit", wires=2)
    print("lightning.qubit (GPU)")
except qml.DeviceError:
    dev = qml.device("default.qubit", wires=2)
    print("default.qubit (CPU)")

class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        # 2 qubits × 2 layers × 3 = 12 parameters
        self.weights = torch.nn.Parameter(torch.randn(12, dtype=torch.float64))
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(x, w):
            qml.AngleEmbedding(x, wires=[0,1])
            qml.StronglyEntanglingLayers(w.reshape(2,2,3), wires=[0,1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        self.circuit = circuit

    def forward(self, x):
        # batched qnode 지원
        return self.circuit(x, self.weights)

# ── 6) HybridCNN ────────────────────────────────────
class HybridCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1,10,3,padding=1); self.bn1 = BatchNorm2d(10)
        self.conv2 = Conv2d(10,20,3,padding=1); self.bn2 = BatchNorm2d(20)
        self.conv3 = Conv2d(20,40,3,padding=1); self.bn3 = BatchNorm2d(40)
        self.drop  = Dropout2d(0.4)
        # spatial:28→14→7→3
        # fc1: 40*3*3=360→112
        self.fc1   = Linear(360,112)
        self.fc2   = Linear(112, 2)
        self.qc    = QuantumCircuit()
        self.out   = Linear(1,   2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x))); x = F.max_pool2d(x,2)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)       # → [B,2] 입력으로 QNN 사용
        q = self.qc(x)        # → [B]
        q = q.view(-1,1)      # → [B,1]
        return self.out(q)    # → [B,2]

# ── 7) Training & Evaluation ────────────────────────
def train_and_eval():
    model = HybridCNN().to(device)
    # 파라미터 카운트 체크
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    q_p     = model.qc.weights.numel()
    spec    = qml.specs(model.qc.circuit)(torch.randn(2), model.qc.weights)
    depth   = spec["resources"].depth
    assert total_p <= 50000 and q_p <= 60 and depth <= 30

    # warm-up
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(1,1,28,28).to(device))
    
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS,
                           steps_per_epoch=len(train_loader))
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    best_acc, patience = 0.0, 0
    for ep in range(1, EPOCHS+1):
        model.train()
        tloss, tpred, ttrue = 0.0, [], []
        for xb,yb in tqdm(train_loader, desc=f"Train {ep}/{EPOCHS}"):
            xb,yb = xb.to(device), yb.to(device)
            xb, ya, yb2, lam = mixup_data(xb,yb)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = lam*loss_fn(logits, ya) + (1-lam)*loss_fn(logits, yb2)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tloss += loss.item()
            preds = logits.argmax(1)
            tpred.extend(preds.cpu().tolist()); ttrue.extend(yb.cpu().tolist())
        tr_acc  = accuracy_score(ttrue,tpred)
        tr_loss = tloss/len(train_loader)

        model.eval()
        vloss, vpred, vtrue = 0.0, [], []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += loss_fn(logits, yb).item()
                preds = logits.argmax(1)
                vpred.extend(preds.cpu().tolist()); vtrue.extend(yb.cpu().tolist())
        val_acc  = accuracy_score(vtrue,vpred)
        val_loss = vloss/len(val_loader)

        print(f"Ep{ep:02d} | TrL {tr_loss:.4f} TrA {tr_acc:.4f} | VaL {val_loss:.4f} VaA {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("⏹ Early stopping")
                break

    # 테스트 추론 & 저장
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    out=[]
    tl = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for xb,_ in tqdm(tl, desc="Test"):
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            out.extend(np.where(preds==1,6,0).tolist())

    fn = os.path.join(SUBMISSION_DIR, f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv")
    np.savetxt(fn, out, fmt='%d')
    print(f"✅ Saved {fn}, Best Val Acc {best_acc:.4f}")

if __name__=="__main__":
    train_and_eval()


# ✅ Saved submission\y_pred_20250804_182540.csv, Best Val Acc 0.8717