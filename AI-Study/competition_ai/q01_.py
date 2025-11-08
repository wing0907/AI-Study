import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import pandas as pd
from datetime import datetime
import pennylane as qml

# ──────────────────────────────────────────────────────────────────────────────
# 0) 경로, Seed, Device 설정
# ──────────────────────────────────────────────────────────────────────────────
# Windows 경로 안전하게
base_path = r'C:\Study25\\competition_ai\\'
os.makedirs(base_path, exist_ok=True)

SEED = 907
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
# 1) 설정값
# ──────────────────────────────────────────────────────────────────────────────
n_qubits           = 5
q_layers           = 3
batch_size         = 64
learning_rate      = 7e-4
weight_decay       = 5e-4
dropout_rate       = 0.5
label_smoothing    = 0.05
cutmix_mixup_prob  = 0.2            # 20% 확률로 CutMix/MixUp 적용
cutmix_or_mixup    = 'mixup'        # 'cutmix' 또는 'mixup'

# ──────────────────────────────────────────────────────────────────────────────
# 2) 데이터 준비 (+ 0 vs 6 필터 & 6->1 매핑)
# ──────────────────────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_full = datasets.FashionMNIST(root=base_path, train=True,  download=True, transform=transform_train)
test_full  = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform_test)

idx_tr = [i for i,(_,l) in enumerate(train_full) if l in (0,6)]
idx_te = [i for i,(_,l) in enumerate(test_full)  if l in (0,6)]
train_ds = Subset(train_full, idx_tr)
test_ds  = Subset(test_full,  idx_te)

# 6 -> 1 매핑
for ds in (train_ds, test_ds):
    t = ds.dataset.targets
    t[t == 6] = 1

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
eval_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Label Smoothing + CutMix/MixUp
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
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch = x.size(0)
    index = torch.randperm(batch).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    b, _, h, w = x.size()
    index = torch.randperm(b).to(x.device)
    y_a, y_b = y, y[index]
    cx, cy   = np.random.randint(w), np.random.randint(h)
    cut_w    = int(w * np.sqrt(1 - lam))
    cut_h    = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam

# ──────────────────────────────────────────────────────────────────────────────
# 4) QNode (PennyLane)
# ──────────────────────────────────────────────────────────────────────────────
def qnode_template(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(x, weights):
        # x: [B, n_qubits]
        # 입력 인코딩
        for i in range(n_qubits):
            qml.RX(x[:, i], wires=i)
        # 층 반복
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            # CZ 링 커넥션
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i+1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

quantum_circuit = qnode_template(n_qubits, q_layers)

# ──────────────────────────────────────────────────────────────────────────────
# 5) 하이브리드 모델
# ──────────────────────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64, 3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,32, 3, padding=1); self.bn4 = nn.BatchNorm2d(32)
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
        x = torch.tanh(x)  # 안정화

        q_out = quantum_circuit(x, self.qp)
        q_out = torch.stack(q_out, dim=1).float()  # [B, n_qubits]

        x = F.relu(self.fc2(q_out))
        return F.log_softmax(self.fc3(x), dim=1)

# ──────────────────────────────────────────────────────────────────────────────
# 6) 스펙 검사(대회 규정 체크)
# ──────────────────────────────────────────────────────────────────────────────
with torch.no_grad():
    spec_model = HybridModel().to(device).eval()
    dummy_x = torch.randn(1, n_qubits)
    dummy_w = spec_model.qp.data
    specs = qml.specs(quantum_circuit)(dummy_x, dummy_w)
    assert specs["num_tape_wires"] <= 8,  "❌ 큐빗 수 초과"
    assert specs["resources"].depth <= 30, "❌ 회로 깊이 초과"
    assert specs["num_trainable_params"] <= 60, "❌ 학습 퀀텀 파라미터 수 초과"

    total_params = sum(p.numel() for p in spec_model.parameters() if p.requires_grad)
    assert total_params <= 50000, "❌ 학습 전체 파라미터 수 초과"
    print("✅ QNN spec OK")
    print(f"✅ Total params: {total_params}")
    del spec_model

# ──────────────────────────────────────────────────────────────────────────────
# 7) 학습 준비
# ──────────────────────────────────────────────────────────────────────────────
model     = HybridModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # verbose 제거
criterion = LabelSmoothingLoss(classes=2, smoothing=label_smoothing)

best_acc   = 0.0
no_improve = 0
patience   = 50
best_path  = os.path.join(base_path, 'best_model.pth')

epochs = 500
for ep in range(1, epochs+1):
    model.train()
    loss_sum, corr, tot = 0.0, 0, 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()

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
        tot  += lbls.size(0)

    train_acc  = corr / tot * 100.0
    train_loss = loss_sum / len(train_loader)

    # 스케줄러(감소시만 로그)
    old_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(train_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != old_lr:
        print(f"[Scheduler] LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

    print(f"Epoch {ep}/{epochs}  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")

    # ✅ 최고 정확도 갱신시에만 저장
    if train_acc > best_acc:
        best_acc = train_acc
        no_improve = 0
        torch.save(model.state_dict(), best_path)
        print(f"  ➡️ New best (Train Acc): {best_acc:.2f}%  |  Saved: {best_path}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("  ⏹ Early stopping (no improvement).")
            break

# ──────────────────────────────────────────────────────────────────────────────
# 8) 평가 & 제출
# ──────────────────────────────────────────────────────────────────────────────
# 최고 성능 시점 모델 로드
model.load_state_dict(torch.load(best_path, map_location=device))
print(f"\n✅ Loaded best model from: {best_path}")
print(f"✅ Best Train Accuracy (during training): {best_acc:.2f}%")

# Eval set 최종 accuracy 출력
model.eval()
corr, tot = 0, 0
with torch.no_grad():
    for imgs, lbls in eval_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs).argmax(1)
        corr += (preds == lbls).sum().item()
        tot  += lbls.size(0)
final_acc = corr / tot * 100.0
print(f"🏁 Final Accuracy on Eval set: {final_acc:.2f}%")

# 제출 CSV
all_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        all_preds.extend(model(imgs).argmax(1).cpu().numpy())

sub = [0 if p == 0 else 6 for p in all_preds]
now = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(base_path, f"submission_{now}.csv")
pd.DataFrame({"y_pred": sub}).to_csv(csv_path, index=False, header=False)
print(f"✅ Saved submission: {csv_path}")
