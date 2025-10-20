import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import Adam, RMSprop, Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import NLLLoss, Module, Conv2d, BatchNorm2d, Dropout2d, Linear
import pennylane as qml
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── 1) Reproducibility ───────────────────────────────
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── 2) Configurations ─────────────────────────────────
BATCH_SIZE      = 256
LR              = 5e-4
EPOCHS          = 200
PATIENCE        = 10
BEST_MODEL_PATH = "best_model.pt"
SUBMISSION_DIR  = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 3) Hyperparameter helper ──────────────────────────
class Hyperparameters:
    def __init__(self, optimizer='adam', activation='relu'):
        self.optimizer_name  = optimizer
        self.activation_name = activation

    def get_optimizer(self, params, lr):
        if   self.optimizer_name == 'adam':    return Adam(params,    lr=lr, weight_decay=5e-4)
        elif self.optimizer_name == 'rmsprop': return RMSprop(params, lr=lr)
        elif self.optimizer_name == 'adadelta':return Adadelta(params)
        else: raise ValueError("Unsupported optimizer")

    def get_activation(self):
        if   self.activation_name == 'relu': return F.relu
        elif self.activation_name == 'elu':  return F.elu
        elif self.activation_name == 'selu': return F.selu
        else: raise ValueError("Unsupported activation")

# ── 4) Data Preparation ───────────────────────────────
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
test_ds    = torchvision.datasets.FashionMNIST(
    "./", train=False, download=True, transform=transform_test)

# Filter only labels 0 vs 6 -> remap 6->1
mask      = (full_train.targets == 0) | (full_train.targets == 6)
full_train.targets[full_train.targets == 6] = 1
binary_ds = Subset(full_train, torch.where(mask)[0])

# train/val split 90/10
val_size   = int(0.1 * len(binary_ds))
train_size = len(binary_ds) - val_size
train_ds, val_ds = random_split(binary_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ── 5) QuantumCircuit 모듈 ───────────────────────────
try:
    dev = qml.device("lightning.qubit", wires=2)
    print("Using lightning.qubit (GPU)")
except qml.DeviceError:
    dev = qml.device("default.qubit", wires=2)
    print("Falling back to default.qubit (CPU)")

class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        # 2 qubits, 2 layers -> 2*2*3=12 trainable params
        self.weights = torch.nn.Parameter(torch.randn(12, dtype=torch.float64))
        
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=[0,1])
            qml.StronglyEntanglingLayers(weights.reshape(2,2,3), wires=[0,1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        self.circuit = circuit

    # **여기를 수정했습니다: forward 함수가 weights를 인자로 받도록 변경**
    def forward(self, x, weights):
        # x: [B,2], weights: [12] -> returns [B]
        # QNode는 배치 입력을 직접 처리할 수 없으므로, 샘플별로 반복 실행
        return torch.stack([self.circuit(xi, weights) for xi in x])

# ── 6) HybridCNN ──────────────────────────────────────
class HybridCNN(Module):
    def __init__(self, activation):
        super().__init__()
        self.act   = activation
        # Conv 레이어 채널 수 확장
        self.conv1 = Conv2d(1, 12, 3, padding=1); self.bn1 = BatchNorm2d(12)
        self.conv2 = Conv2d(12,24, 3, padding=1); self.bn2 = BatchNorm2d(24)
        self.conv3 = Conv2d(24,48, 3, padding=1); self.bn3 = BatchNorm2d(48)
        self.drop  = Dropout2d(0.4)
        # FC 레이어 노드 수 확장
        self.fc1   = Linear(48*3*3, 64)
        self.fc2   = Linear(64,   12) # QNN 입력 크기 12에 맞춤
        self.qc    = QuantumCircuit()
        self.out   = Linear(1,    2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x))); x = F.max_pool2d(x,2)
        x = self.act(self.bn2(self.conv2(x))); x = F.max_pool2d(x,2)
        x = self.act(self.bn3(self.conv3(x))); x = F.max_pool2d(x,2)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)            # [B,12] ready for QNN

        # **여기서 QNN 입력 데이터를 2개만 사용하도록 수정했습니다**
        qnn_input = x[:, :2] # 12개의 특징 중 앞 2개만 사용

        # **여기를 수정했습니다: qc의 forward 함수를 호출하며 weights를 명시적으로 전달**
        q_out = self.qc(qnn_input, self.qc.weights) # [B]
        q_out = q_out.view(-1,1)   # [B,1]
        return self.out(q_out)     # [B,2]


# ── 7) Training / Evaluation ─────────────────────────
def run_experiment(hp):
    print(f"\n=== Experiment: optim={hp.optimizer_name}, act={hp.activation_name} ===")
    
    # QNN 스펙 체크
    qc = QuantumCircuit()
    # **qml.specs의 입력 데이터 크기를 2로 수정**
    specs = qml.specs(qc.circuit)(torch.randn(2), qc.weights)
    print(f"Q-params: {qc.weights.numel()}, wires: {specs['num_tape_wires']}, depth: {specs['resources'].depth}")
    assert qc.weights.numel() <= 60
    assert specs['num_tape_wires'] <= 8
    assert specs['resources'].depth <= 30

    model = HybridCNN(hp.get_activation()).to(device)
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {tot}")
    assert tot <= 50000

    # WARMUP: QNode 컴파일 지연 방지
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1,1,28,28).to(device)
        _ = model(dummy)

    opt = hp.get_optimizer(model.parameters(), LR)
    sch = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loss_fn = NLLLoss()

    best_acc, wait = 0.0, 0
    for ep in range(1, EPOCHS+1):
        # Training
        model.train()
        tloss, tpred, ttrue = 0.0, [], []
        for xb,yb in tqdm(train_loader, desc=f"Train {ep}/{EPOCHS}"):
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(F.log_softmax(logits,1), yb)
            loss.backward(); opt.step()
            tloss += loss.item()
            preds = logits.argmax(1)
            tpred.extend(preds.cpu().numpy()); ttrue.extend(yb.cpu().numpy())
        train_acc = accuracy_score(ttrue, tpred)
        train_loss= tloss/len(train_loader)

        # Validation
        model.eval()
        vloss, vpred, vtrue = 0.0, [], []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                vloss += loss_fn(F.log_softmax(logits,1), yb).item()
                preds = logits.argmax(1)
                vpred.extend(preds.cpu().numpy()); vtrue.extend(yb.cpu().numpy())
        val_acc  = accuracy_score(vtrue, vpred)
        val_loss = vloss/len(val_loader)

        print(f"Ep{ep} | TrLoss {train_loss:.4f} Acc {train_acc:.4f} | ValLoss {val_loss:.4f} Acc {val_acc:.4f}")
        sch.step(val_loss)

        if val_acc > best_acc:
            best_acc, wait = val_acc, 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # 최종 추론 및 저장
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    out=[]
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for xb,_ in tqdm(test_loader, desc="Test"): 
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            out.extend(np.where(preds==1,6,0).tolist())
    fname = os.path.join(SUBMISSION_DIR, f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv")
    np.savetxt(fname, out, fmt='%d')
    print(f"▶ Saved {fname} (Best Val Acc {best_acc:.4f})\n")

# ── 8) Run all experiments ───────────────────────────
if __name__=="__main__":
    exps = [
        Hyperparameters('adam',    'relu'),
        Hyperparameters('adam',    'elu'),
        Hyperparameters('adam',    'selu'),
        Hyperparameters('rmsprop', 'relu'),
        Hyperparameters('adadelta','relu'),
    ]
    for hp in exps:
        run_experiment(hp)

# === Experiment: optim=adam, act=elu ===
# ▶ Saved submission\y_pred_20250804_140831.csv (Best Val Acc 0.8742) Ep36 | TrLoss 0.3682 Acc 0.8570 | ValLoss 0.3528 Acc 0.8742  PUBLIC 0.869

# === Experiment: optim=adam, act=selu ===
# ▶ Saved submission\y_pred_20250804_150150.csv (Best Val Acc 0.8825) Ep45 | TrLoss 0.3762 Acc 0.8646 | ValLoss 0.3640 Acc 0.8825  PUBLIC 0.872

# === Experiment: optim=rmsprop, act=relu ===
# ▶ Saved submission\y_pred_20250804_152546.csv (Best Val Acc 0.8633) Ep14 | TrLoss 0.4100 Acc 0.8407 | ValLoss 0.3832 Acc 0.8633

# === Experiment: optim=adadelta, act=relu ===
# ▶ Saved submission\y_pred_20250804_161718.csv (Best Val Acc 0.8867) Ep42 | TrLoss 0.2712 Acc 0.8768 | ValLoss 0.2838 Acc 0.8867  PUBLIC 0.8705