import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
import pennylane as qml
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1) Reproducibility
SEED = 991
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2) Configuration
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 50
PATIENCE = 10
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3) Data Augmentation & Split
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_train = torchvision.datasets.FashionMNIST(
    './', train=True, download=True, transform=transform_train
)
test_ds = torchvision.datasets.FashionMNIST(
    './', train=False, download=True, transform=transform_test
)
# filter 0 vs 6, remap 6->1
mask = (full_train.targets == 0) | (full_train.targets == 6)
full_train.targets[full_train.targets == 6] = 1
binary_ds = Subset(full_train, torch.where(mask)[0])
# split
val_size = int(0.1 * len(binary_ds))
train_size = len(binary_ds) - val_size
train_ds, val_ds = random_split(binary_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 4) MixUp & CutMix utilities

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y, y[idx], lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# 5) Quantum Circuit (4 qubits, 2 layers)
try:
    dev = qml.device('lightning.qubit', wires=4)
    print('Using lightning.qubit (GPU)')
except:
    dev = qml.device('default.qubit', wires=4)
    print('Using default.qubit (CPU)')

class QuantumCircuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2 * 4 * 3, dtype=torch.float64))
        @qml.qnode(dev, interface='torch', diff_method='adjoint')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=list(range(4)))
            W = weights.reshape(2, 4, 3)
            qml.StronglyEntanglingLayers(W, wires=list(range(4)))
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        self.circuit = circuit
    def forward(self, x):
        # batch-capable
        return self.circuit(x, self.weights)

# 6) Hybrid Model with GAP
class HybridCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.act = activation
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.drop  = nn.Dropout2d(0.4)
        self.qc    = QuantumCircuit()
        self.project = nn.Linear(64, 4)
        self.classifier = nn.Linear(1, 2)
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.act(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.act(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.act(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)   # now 1x1 spatial
        x = x.view(x.size(0), -1)  # [B,64]
        x = self.drop(x)
        embed = self.project(x).float()  # [B,4]
        q_out = self.qc(embed).view(-1,1).float()  # [B,1]
        return self.classifier(q_out)

# 7) Training & Validation

def run_experiment():
    model = HybridCNN(F.relu).to(device)
    # specs
    specs = qml.specs(model.qc.circuit)(torch.randn(4), torch.randn(24))
    assert specs['num_trainable_params'] <= 60
    assert specs['resources'].num_wires <= 8
    assert specs['resources'].depth <= 30
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert tot <= 50000
    print(f"Total params: {tot}, Q-params: {specs['num_trainable_params']}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=LR*10, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    best_acc, wait = 0, 0

    # warm-up QNode
    with torch.no_grad(): _ = model(torch.randn(1,1,28,28).to(device))

    for ep in range(1, EPOCHS+1):
        model.train(); tloss=0; tpred=[]; ttrue=[]
        for xb, yb in tqdm(train_loader, desc=f"Train {ep}/{EPOCHS}"):
            xb, yb = xb.to(device), yb.to(device)
            # MixUp half the time
            if random.random()<0.5:
                xb, ya, yb2, lam = mixup_data(xb, yb)
                logits = model(xb)
                loss = lam*loss_fn(logits, ya) + (1-lam)*loss_fn(logits, yb2)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            tloss += loss.item()
            preds = logits.argmax(1)
            tpred.extend(preds.cpu().numpy()); ttrue.extend(yb.cpu().numpy())
        train_acc = accuracy_score(ttrue, tpred)
        # validation
        model.eval(); vpred=[]; vtrue=[]; vloss=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += loss_fn(logits, yb).item()
                p = logits.argmax(1); vpred.extend(p.cpu().numpy()); vtrue.extend(yb.cpu().numpy())
        val_acc = accuracy_score(vtrue, vpred)
        print(f"Ep{ep} | TrAcc{train_acc:.4f} | ValAcc{val_acc:.4f}")
        if val_acc>best_acc: best_acc=val_acc; wait=0; torch.save(model.state_dict(), BEST_MODEL_PATH)
        else: wait+=1;
