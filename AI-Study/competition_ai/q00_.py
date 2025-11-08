import multiprocessing
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
import pandas 
import optuna

# ──────────────────────────────────────────────────────────────────────────────
# 0️⃣ 경로, Seed, Device 설정
# ──────────────────────────────────────────────────────────────────────────────
base_path = 'C:\Study25\competition_ai\\'
os.makedirs(base_path, exist_ok=True)

# ─────────────────────────────────────────────
# ✅ 설정값
# ─────────────────────────────────────────────
n_qubits = 5
q_layers = 3
batch_size = 64
learning_rate = 0.0007
weight_decay = 0.0005
dropout_rate = 0.5
label_smoothing = 0.05
cutmix_mixup_prob = 0.2

SEED = 1856
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ 양자 회로 정의
# ──────────────────────────────────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=n_qubits)

def qnode_template(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(x, weights):
        for i in range(n_qubits):
            qml.RX(x[:, i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i+1)%n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

quantum_circuit = qnode_template(n_qubits, q_layers)


# 데이터 전처리 및 0 vs 6 필터링

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_full = datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform_train)
test_full  = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform_test)
idx_tr = [i for i,(_,l) in enumerate(train_full) if l in (0,6)]
idx_te = [i for i,(_,l) in enumerate(test_full)  if l in (0,6)]
train_ds = Subset(train_full, idx_tr)
test_ds  = Subset(test_full,  idx_te)
for ds in (train_ds, test_ds):
    targets = ds.dataset.targets
    targets[targets == 6] = 1

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ Label Smoothing + CutMix/MixUp
# ──────────────────────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.03):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam

cutmix_or_mixup = 'mixup'  # 'cutmix' 또는 'mixup'


# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ HybridModel 정의 (채널 축소 & Dropout ↑)
# ──────────────────────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(32)

        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(dropout_rate)
        self.avg   = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1   = nn.Linear(32, n_qubits)
        self.norm  = nn.LayerNorm(n_qubits)

        self.qp    = nn.Parameter(torch.randn(q_layers, n_qubits, 2))

        self.fc2   = nn.Linear(n_qubits, 32)
        self.fc3   = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x))); x = self.pool(x)
        x = self.drop(x)
        x = self.avg(x).view(x.size(0), -1)

        x = self.fc1(x)
        x = self.norm(x)
        x = torch.tanh(x)  # ⬅️ 안정화

        q_out = quantum_circuit(x, self.qp)
        q_out = torch.stack(q_out, dim=1).float()

        x = self.fc2(q_out)
        x = F.relu(x)
        return F.log_softmax(self.fc3(x), dim=1)

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ 메인 실행: 로더, 스펙 검사, 학습, 저장, 평가, 제출
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    eval_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    # QNN 스펙 검사
    spec_model = HybridModel().to(device).eval()
    dummy_x = torch.randn(1,5)
    dummy_w = spec_model.qp.data
    specs = qml.specs(quantum_circuit)(dummy_x, dummy_w)
    assert specs["num_tape_wires"] <= 8 and specs["resources"].depth <= 30 and specs["num_trainable_params"] <= 60
    print("✅ QNN spec OK")
    total_p = sum(p.numel() for p in spec_model.parameters() if p.requires_grad)
    assert total_p <= 50000
    print(f"✅ Total params: {total_p}")
    del spec_model

    # 모델/옵티마이저/스케줄러/로스 설정
    model = HybridModel().to(device)

    # 일회성 스펙 및 파라미터 검사 (i==0 상황처럼 한 번만 실행)
    dummy_x2   = torch.randn(1, 5)                   # circuit 입력 더미
    dummy_w2   = model.qp.data                       # 실제 학습 파라미터 사용
    specs2     = qml.specs(quantum_circuit)(dummy_x2, dummy_w2)
    assert specs2["num_tape_wires"]      <= 8,    "❌ 큐빗 수 초과"
    assert specs2["resources"].depth     <= 30,   "❌ 회로 깊이 초과"
    assert specs2["num_trainable_params"]<= 60,   "❌ 학습 퀀텀 파라미터 수 초과"
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 50000, "❌ 학습 전체 파라미터 수 초과"
    print(f"Total Trainable Parameters: {total_params}")
    print("\n✅ 모든 회로 및 모델 제약 통과 — 학습을 계속합니다\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )
    criterion = LabelSmoothingLoss(classes=2, smoothing=label_smoothing)

    best_acc = 0.0
    no_improve = 0
    patience = 50
    best_path = os.path.join(base_path, 'best_model.pth')

    epochs = 500
    for ep in range(1, epochs+1):
        model.train()
        loss_sum, corr, tot = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()

            # CutMix or MixUp 적용
            if np.random.rand() < cutmix_mixup_prob:
                if cutmix_or_mixup == 'cutmix':
                    imgs_mix, targets_a, targets_b, lam = cutmix_data(imgs, lbls)
                elif cutmix_or_mixup == 'mixup':
                    imgs_mix, targets_a, targets_b, lam = mixup_data(imgs, lbls)
                else:
                    raise ValueError("cutmix_or_mixup must be 'cutmix' or 'mixup'")

                outputs = model(imgs_mix)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            preds = outputs.argmax(1)
            corr += (preds == lbls).sum().item()
            tot += lbls.size(0)

        train_acc = corr / tot * 100
        train_loss = loss_sum / len(train_loader)
        print(f"Epoch {ep}/{epochs}  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
        scheduler.step(train_loss)

        if train_acc > best_acc:
            best_acc = train_acc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ➡️ New best: {best_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("  ⏹ Early stopping")
                break

    # 최종 모델 로드 & 저장
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(base_path, f"model_{now}_{best_acc:.4f}.pth")
    model.load_state_dict(torch.load(best_path))
    torch.save(model.state_dict(), final_path)
    print(f"✅ Saved final model: {final_path}")

    # 평가
    model.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for imgs, lbls in eval_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(1)
            corr += (preds==lbls).sum().item()
            tot += lbls.size(0)
    test_acc = corr/tot*100
    print(f"✅ Test 0 vs 6 Accuracy: {test_acc:.2f}%")

    # 제출 CSV
    all_preds = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            all_preds.extend(model(imgs).argmax(1).cpu().numpy())
    sub = [0 if p==0 else 6 for p in all_preds]
    df = pd.DataFrame({"y_pred": sub})
    csv_path = os.path.join(base_path, f"submission_{now}.csv")
    df.to_csv(csv_path, index=False, header=False)
    print(f"✅ Saved submission: {csv_path}")

# seed 222
# ✅ Saved final model: C:\Study25\competition_ai\model_20250808_181700_96.2500.pth
# ✅ Test 0 vs 6 Accuracy: 89.55%
# ✅ Saved submission: C:\Study25\competition_ai\submission_20250808_181700.csv

# seed 6054
# ✅ Saved final model: C:\Study25\competition_ai\model_20250808_185610_97.0833.pth
# ✅ Test 0 vs 6 Accuracy: 89.05%
# ✅ Saved submission: C:\Study25\competition_ai\submission_20250808_185610.csv

# seed 1856
# ✅ Saved final model: C:\Study25\competition_ai\model_20250808_193729_96.2750.pth
# ✅ Test 0 vs 6 Accuracy: 88.95%
# ✅ Saved submission: C:\Study25\competition_ai\submission_20250808_193729.csv
