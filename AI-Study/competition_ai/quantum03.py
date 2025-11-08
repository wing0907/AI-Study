import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import NLLLoss, Module, Conv2d, BatchNorm2d, Dropout2d, Linear
from torch.nn.parameter import Parameter
# hybrid_extractor stub for missing package
try:
    from hybrid_extractor import HybridExtractor
except ImportError:
    class HybridExtractor:
        def __init__(self, *args, **kwargs):
            print("Warning: hybrid_extractor not installed. Using stub HybridExtractor.")
        def extract(self, *args, **kwargs):
            raise NotImplementedError("HybridExtractor.extract() not implemented. Provide hybrid_extractor.py in project.")
import pennylane as qml
from tqdm import tqdm
from datetime import datetime

# ── Reproducibility / Seed ─────────────────────────
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Configurations ─────────────────────────────────
BATCH_SIZE = 256
LR = 5e-4
EPOCHS = 200
PATIENCE = 10
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ── Device Setup ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Preparation w/ Augmentation & Validation Split ─────────────────
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
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
# Filter labels 0 vs 6, map 6->1
mask_train = (full_train.targets == 0) | (full_train.targets == 6)
full_train.targets[full_train.targets == 6] = 1
indices = torch.where(mask_train)[0]
binary_ds = Subset(full_train, indices)
# Split train/val
val_size = int(0.1 * len(binary_ds))
train_size = len(binary_ds) - val_size
train_ds, val_ds = random_split(binary_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── Quantum Circuit (2 qubits, 2 layers) ─────────────────────────────
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=2)
        self.params = Parameter(torch.randn(12, dtype=torch.float64), requires_grad=True)
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.AngleEmbedding(x, wires=[0,1])
            # two entangling layers
            for l in range(2):
                qml.StronglyEntanglingLayers(
                    self.params.reshape(2,2,3), wires=[0,1]
                )
            return qml.expval(self.obs)
        self.circuit = circuit
    def forward(self, x):
        return self.circuit(x)

# ── Hybrid Model: CNN + QNN ───────────────────────────────────────────
class HybridCNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 16, 3, padding=1)
        self.bn1   = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.bn2   = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.bn3   = BatchNorm2d(64)
        self.dropout = Dropout2d(0.4)
        # after 3 pools: 28->14->7->3
        self.fc1 = Linear(64*3*3, 128)
        self.fc2 = Linear(128, 2)  # 2-dim for quantum input
        self.qnn = QuantumCircuit()
        self.final = Linear(1, 2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x,2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # QNN
        q_out = torch.stack([self.qnn(vec) for vec in x]).view(-1,1).float()
        logits = self.final(q_out)
        return logits

# ── Training with Validation & Scheduler ─────────────────────────────
model = HybridCNN().to(device)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
loss_fn = NLLLoss()
best_val = float('inf')
patience_ctr = 0

for epoch in range(1, EPOCHS+1):
    scheduler.step()
    model.train()
    train_loss = 0
    for data,target in tqdm(train_loader, desc=f"Train {epoch}"):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(F.log_softmax(logits,dim=1), target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for data,target in val_loader:
            data,target = data.to(device), target.to(device)
            logits = model(data)
            val_loss += loss_fn(F.log_softmax(logits,dim=1), target).item()
            preds = logits.argmax(dim=1)
            correct += (preds==target).sum().item()
            total += target.size(0)
    val_loss /= len(val_loader)
    val_acc = correct/total
    print(f"Epoch {epoch} Train Loss {train_loss:.4f} Val Loss {val_loss:.4f} Val Acc {val_acc:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        patience_ctr = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Saved best model")
    else:
        patience_ctr +=1
        if patience_ctr>=PATIENCE:
            print("Early stopping")
            break

# ── Final Inference & Submission ─────────────────────────────────────
# Use full test ds without filter
test_full = torchvision.datasets.FashionMNIST(
    "./", train=False, download=True,
    transform=transform_test)
loader_full = DataLoader(test_full, batch_size=BATCH_SIZE, shuffle=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

all_preds=[]
with torch.no_grad():
    for data,_ in tqdm(loader_full, desc="Test Inf"):
        data = data.to(device)
        logits = model(data)
        preds = logits.argmax(dim=1).cpu().numpy()
        # map back: 1->6, 0->0
        y = np.where(preds==1,6,0)
        all_preds.extend(y.tolist())

assert len(all_preds)==len(test_full)
filename=f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
np.savetxt(filename, all_preds, fmt='%d')
print(f"Saved submission {filename}")

# seed 6054
# Epoch 100 Train Loss 0.2876 Val Loss 0.3142 Val Acc 0.8800
# Saved submission y_pred_20250728_210952.csv

# Saved submission y_pred_20250729_063717.csv

model = HybridCNN().to(device)
# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")