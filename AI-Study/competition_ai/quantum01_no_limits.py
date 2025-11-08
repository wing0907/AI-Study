"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import NLLLoss
from tqdm import tqdm
import pennylane as qml
from datetime import datetime

# ── Configurations ─────────────────────────────────
SEED = 321
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 100
PATIENCE = 10
VAL_RATIO = 0.2
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR = "submission"

os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ── Utility Functions ─────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = torchvision.datasets.FashionMNIST(
        "./", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST(
        "./", train=False, download=True, transform=transform)

    # Filter 0 vs 6
    mask = (full_train.targets == 0) | (full_train.targets == 6)
    indices = torch.where(mask)[0]
    full_train.targets[full_train.targets == 6] = 1
    binary_train = Subset(full_train, indices)

    # Train/Val split
    val_size = int(len(binary_train) * VAL_RATIO)
    train_size = len(binary_train) - val_size
    train_ds, val_ds = random_split(binary_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def save_submission(y_pred):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"y_pred_{now}.csv"
    path = os.path.join(SUBMISSION_DIR, filename)
    np.savetxt(path, y_pred, fmt='%d', delimiter=",")
    return path

# ── Model Definition ──────────────────────────────
class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # CNN part
        self.conv1 = torch.nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(2, 16, kernel_size=5)
        self.dropout = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 2)
        self.fc3 = torch.nn.Linear(1, 1)

        # Quantum part
        self.q_device = qml.device("default.qubit", wires=2)
        self.q_params = torch.nn.Parameter(torch.rand(8, dtype=torch.float64))
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

        @qml.qnode(self.q_device, interface="torch", diff_method="backprop")
        def circuit(x):
            # Encoding
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            # Variational layers
            for i in range(4):
                qml.RY(self.q_params[2*i],   wires=0)
                qml.RX(self.q_params[2*i+1], wires=1)
                if i % 2 == 0:
                    qml.CNOT(wires=[0, 1])
                else:
                    qml.CNOT(wires=[1, 0])
            return qml.expval(self.obs)

        self.qnn = circuit

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # Flatten conv output: maintain batch dimension
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Quantum layer across batch
        q_out = torch.stack([self.qnn(sample) for sample in x])  # shape [batch]
        q_out = q_out.view(-1, 1).float()
        x = self.fc3(q_out)
        return F.log_softmax(torch.cat((x, 1 - x), dim=-1), dim=-1)

# ── Training & Evaluation ─────────────────────────
def train_model(model, train_loader, val_loader, device):
    set_seed()
    optimizer = Adam(model.parameters(), lr=LR)
    loss_func = NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=SCHEDULER_PATIENCE,
                                  factor=SCHEDULER_FACTOR)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            data, target = data.to(device), target.view(-1).to(device)
            optimizer.zero_grad()
            loss = loss_func(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_val = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.view(-1).to(device)
                total_val += loss_func(model(data), target).item()
        avg_val = total_val / len(val_loader)
        scheduler.step(avg_val)
        print(f"[{epoch+1}/{EPOCHS}] train_loss: {avg_train:.4f}, val_loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

# ── Main ──────────────────────────────────────────
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders()
    model = BinaryClassifier().to(device)
    train_model(model, train_loader, val_loader, device)

    # Inference & submission
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    preds = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            pred = model(data).argmax(dim=1)
            preds.extend(pred.cpu().numpy())
    y_pred = np.array(preds, dtype=int)
    y_pred = np.where(y_pred == 1, 6, y_pred)
    path = save_submission(y_pred)
    print(f"✅ Saved submission: {path}")

if __name__ == "__main__":
    main()
"""

# ✅ Saved submission: submission\y_pred_20250722_210605.csv

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import CrossEntropyLoss
import pennylane as qml
from datetime import datetime
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings('ignore')

# Ensure default dtype for Torch and PennyLane
torch.set_default_dtype(torch.float64)

# 1) Reproducibility
# ------------------
def set_seed(seed=6054):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 2) Quantum Circuit Definition
# ------------------------------
class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4):
        super().__init__()
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # parameters as torch Parameter, double precision
        self.params = nn.Parameter(
            torch.randn(n_layers * n_qubits * 3, dtype=torch.float64)
        )

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(x, weights):
            # cast input to double
            x = x.to(torch.float64)
            qml.AngleEmbedding(x, wires=list(range(n_qubits)))
            W = weights.reshape(n_layers, n_qubits, 3)
            qml.StronglyEntanglingLayers(W, wires=list(range(n_qubits)))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x):
        return self.circuit(x.double(), self.params)

# 3) Hybrid ResNet + QNN Model
# -----------------------------
class HybridResNetQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        # project to QNN input dim (n_qubits)
        self.proj = nn.Linear(512, 4)
        self.qnn  = QuantumCircuit(n_qubits=4, n_layers=4)
        self.cls  = nn.Linear(1, 2)

    def forward(self, x):
        feats = self.backbone(x)
        feats = F.relu(self.proj(feats))
        # ensure double for quantum
        feats = feats.double()
        q_out = torch.stack([self.qnn(f) for f in feats])
        q_out = q_out.view(-1,1)
        return self.cls(q_out)

# 4) Main
if __name__ == '__main__':
    freeze_support()
    set_seed(6054)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    full = torchvision.datasets.FashionMNIST(
        './', train=True, download=True, transform=transform_train)
    test_ds = torchvision.datasets.FashionMNIST(
        './', train=False, download=True, transform=transform_test)
    mask = (full.targets == 0) | (full.targets == 6)
    full.targets[full.targets == 6] = 1
    subset = Subset(full, torch.where(mask)[0])
    val_size = int(0.1 * len(subset))
    train_size = len(subset) - val_size
    train_ds, val_ds = random_split(subset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model = HybridResNetQNN().to(device)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = OneCycleLR(optim, max_lr=1e-2, epochs=30, steps_per_epoch=len(train_loader))
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

    best_val = float('inf'); patience = 0; best_path = None
    for ep in range(30):
        model.train(); total_loss = 0.0
        for data, tgt in train_loader:
            data, tgt = data.to(device), tgt.to(device)
            optim.zero_grad()
            logits = model(data)
            loss = loss_fn(logits, tgt)
            loss.backward()
            optim.step(); sched.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1} Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval(); val_loss=0.0; correct=0; total=0
        with torch.no_grad():
            for data, tgt in val_loader:
                data, tgt = data.to(device), tgt.to(device)
                logits = model(data)
                val_loss += loss_fn(logits, tgt).item()
                preds = logits.argmax(dim=1)
                correct += (preds == tgt).sum().item(); total += tgt.size(0)
        val_loss /= len(val_loader); val_acc = correct/total
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss; patience=0
            best_path = f"best_qnn_epoch{ep+1}.pt"
            torch.save(model.state_dict(), best_path)
            print(f"Saved {best_path}")
        else:
            patience +=1
            if patience >= 5:
                print("Early stopping"); break

    model.load_state_dict(torch.load(best_path))
    model.eval(); results=[]
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            preds = model(data).argmax(dim=1).cpu().numpy()
            results.extend(np.where(preds == 1, 6, 0).tolist())
    assert len(results) == len(test_ds)
    out_name = f"y_pred_resnet_qnn_{datetime.now():%Y%m%d_%H%M%S}.csv"
    np.savetxt(out_name, results, fmt='%d')
    print(f"Saved submission {out_name}")
