##############################
# 전체 코드 (모델 저장 + seed 고정)
##############################

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

##############################
# 0️⃣ 경로 설정
##############################
base_path = './Study25/_data/AIFactory/Quantum/'
os.makedirs(base_path, exist_ok=True)

##############################
# 1️⃣ Seed 고정
##############################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

##############################
# 2️⃣ 디바이스 선언
##############################
dev = qml.device("default.qubit", wires=5)

##############################
# 3️⃣ QNN 회로 정의
##############################
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(5):
        qml.RX(inputs[i % 2], wires=i)
        qml.RY(weights[i % 2], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

##############################
# 4️⃣ 데이터 준비
##############################
transform = transforms.Compose([
    transforms.ToTensor()
])

# 전체 데이터셋 로드
train_dataset = datasets.FashionMNIST(root=base_path, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=base_path, train=False, download=True, transform=transform)

# ✅ 0/6 class만 선택한 train subset 만들기
indices = [i for i, (_, label) in enumerate(train_dataset) if label in [0, 6]]
train_dataset = Subset(train_dataset, indices)

# ✅ relabel: 6 → 1
for idx in train_dataset.indices:
    orig_label = train_dataset.dataset.targets[idx]
    if orig_label == 6:
        train_dataset.dataset.targets[idx] = 1
    else:
        train_dataset.dataset.targets[idx] = 0

# ✅ DataLoader 구성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

##############################
# 5️⃣ 모델 정의
##############################
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 3 * 3, 32)  # Feature map size 3x3 assumed
        self.q_params = nn.Parameter(torch.rand(8))  # QNN param 8개 이상 60개 이하
        self.fc2 = nn.Linear(1, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        q_out = torch.stack([quantum_circuit(x[i], self.q_params) for i in range(x.shape[0])])
        q_out = q_out.unsqueeze(1).to(torch.float32)
        x = self.fc2(q_out)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

##############################
# 6️⃣ 규격 검사
##############################
dummy_x = torch.tensor([0.1, 0.2])
specs = qml.specs(quantum_circuit)(dummy_x, torch.tensor([0.1, 0.2]))
assert specs["num_tape_wires"] <= 8
assert specs["resources"].depth <= 30
assert specs["num_trainable_params"] <= 60
print("✅ 규격 검사 통과")

##############################
# 7️⃣ 학습
##############################
model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.NLLLoss()

best_loss = float('inf')  # ⭐ best loss 초기화

best_acc = 0.0  # ⭐ best accuracy 초기화

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # ⭐ accuracy 계산
        preds = output.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total * 100

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")

    # ⭐ best acc 기준으로 모델 저장
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_model_path = os.path.join(base_path, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated at epoch {epoch+1} with acc {avg_acc:.2f}%")

##############################
# 8️⃣ 모델 저장 (마지막 epoch 기준)
##############################
now = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(base_path, f'model_{now}_{best_acc:.4f}.pth')
torch.save(model.state_dict(), model_path)
print(f"✅ 마지막 모델 저장 완료: {model_path}")

##############################
# 9️⃣ 추론 및 제출 파일 저장
##############################
# ⭐ best model load 추가
best_model_path = os.path.join(base_path, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path))
print(f"✅ Best model loaded for inference from: {best_model_path}")

model.eval()
preds = []
with torch.no_grad():
    for images, _ in test_loader:
        output = model(images)
        pred = output.argmax(dim=1)
        preds.extend(pred.cpu().numpy())

preds = [0 if p == 0 else 6 for p in preds]
print(f"제출용 예측값 길이: {len(preds)}")

csv_filename = f"{base_path}y_pred_{now}_{best_acc:.4f}.csv"
df = pd.DataFrame({"y_pred": preds})
df.to_csv(csv_filename, index=False, header=False)
print(f"✅ 결과 저장 완료: {csv_filename}")

# 20250714_194927 Loss : 0.4084
# 20250714_201118 Loss : 0.2110
# 20250714_204906 Loss : 0.0478
# 20250715_101019 Loss : 0.2141
# 20250715_104240 Loss : 0.0440
# 20250715_111651 Loss : 0.0453
# 20250716_003747 
