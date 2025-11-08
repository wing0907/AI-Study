# # # quantum_classifier.py

# # import torch
# # import pennylane as qml
# # import torch.nn.functional as F
# # import numpy as np
# # from torch import cat
# # from torch.nn import Module, Linear, Conv2d, Dropout2d, NLLLoss, BatchNorm2d
# # from torch.nn.parameter import Parameter
# # from torch.optim import Adam
# # from torch.utils.data import DataLoader, Subset
# # import torchvision
# # from torchvision import transforms
# # from tqdm import tqdm
# # from datetime import datetime

# # # 1. Device & 기본 설정
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # torch.set_default_dtype(torch.float64)

# # # 2. FashionMNIST 0 vs 6만 선택
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5,), (0.5,))
# # ])
# # train_ds = torchvision.datasets.FashionMNIST("./", train=True, download=True, transform=transform)
# # test_ds = torchvision.datasets.FashionMNIST("./", train=False, download=True, transform=transform)

# # train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
# # train_idx = torch.where(train_mask)[0]
# # train_ds.targets[train_ds.targets == 6] = 1
# # binary_train_ds = Subset(train_ds, train_idx)
# # train_loader = DataLoader(binary_train_ds, batch_size=1, shuffle=True)

# # # 3. Quantum + CNN Classifier
# # class BinaryClassifier(Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.conv1 = Conv2d(1, 8, kernel_size=5)
# #         self.bn1 = BatchNorm2d(8)
# #         self.conv2 = Conv2d(8, 32, kernel_size=5)
# #         self.bn2 = BatchNorm2d(32)
# #         self.dropout = Dropout2d(0.3)
# #         self.fc1 = Linear(512, 64)
# #         self.fc2 = Linear(64, 2)
# #         self.fc3 = Linear(1, 1)

# #         self.q_device = qml.device("default.qubit", wires=2)
# #         self.qnn_params = Parameter(torch.rand(8, dtype=torch.float64), requires_grad=True)
# #         self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

# #         @qml.qnode(self.q_device, interface="torch")
# #         def circuit(x):
# #             qml.H(0)
# #             qml.H(1)
# #             qml.RZ(2.*x[0], 0)
# #             qml.RZ(2.*x[1], 0)
# #             qml.CNOT([0, 1])
# #             qml.RZ(2.*(torch.pi - x[0])*(torch.pi - x[1]), 1)
# #             qml.CNOT([0, 1])
# #             qml.RY(2.*self.qnn_params[0], 0)
# #             qml.RY(2.*self.qnn_params[1], 1)
# #             qml.CNOT([0, 1])
# #             qml.RY(2.*self.qnn_params[2], 0)
# #             qml.RY(2.*self.qnn_params[3], 1)
# #             qml.CNOT([1, 0])
# #             qml.RY(2.*self.qnn_params[4], 0)
# #             qml.RY(2.*self.qnn_params[5], 1)
# #             qml.CNOT([0, 1])
# #             qml.RY(2.*self.qnn_params[6], 0)
# #             qml.RY(2.*self.qnn_params[7], 1)
# #             return qml.expval(self.obs)

# #         self.qnn = circuit

# #     def forward(self, x):
# #         x = F.relu(self.conv1(x))
# #         x = F.max_pool2d(x, 2)
# #         x = F.relu(self.conv2(x))
# #         x = F.max_pool2d(x, 2)
# #         x = self.dropout(x)
# #         x = x.view(-1)
# #         x = F.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         x = self.qnn(x).view(1,)
# #         x = self.fc3(x)
# #         return F.log_softmax(cat((x, 1 - x), -1), -1)

# # # 4. 모델 초기화
# # bc = BinaryClassifier().to(device)

# # # 5. 회로 제약 검증
# # total_params = sum(p.numel() for p in bc.parameters() if p.requires_grad)
# # dummy_x = torch.tensor([0.0, 0.0], dtype=torch.float64)
# # specs = qml.specs(bc.qnn)(dummy_x)
# # assert specs["num_tape_wires"] <= 8
# # assert specs['resources'].depth <= 30
# # assert specs["num_trainable_params"] <= 60
# # assert total_params <= 50000
# # print("✅ 회로 제약 통과 — 학습 시작")

# # # 6. 학습
# # optimizer = Adam(bc.parameters(), lr=0.0001)
# # loss_func = NLLLoss()
# # epochs = 10
# # loss_history = []

# # bc.train()
# # for epoch in range(epochs):
# #     epoch_bar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
# #     total_loss = []
# #     for bidx, (data, target) in zip(epoch_bar, train_loader):
# #         data, target = data.to(device), target.to(device)
# #         optimizer.zero_grad(set_to_none=True)
# #         output = bc(data)
# #         loss = loss_func(output, target.squeeze())
# #         total_loss.append(loss.item())
# #         loss.backward()
# #         optimizer.step()
# #         if bidx % 100 == 0:
# #             epoch_bar.set_postfix(batch=bidx, loss=f"{loss.item():.4f}")
# #     avg_loss = sum(total_loss) / len(total_loss)
# #     loss_history.append(avg_loss)
# #     print(f"Training [{100.0 * (epoch+1)/epochs:.0f}%] Loss: {avg_loss:.4f}")

# # # 7. 추론
# # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
# # bc.eval()
# # all_preds, all_targets = [], []

# # with torch.no_grad():
# #     for data, target in tqdm(test_loader, desc="Inference", total=len(test_loader), leave=False):
# #         data, target = data.to(device), target.to(device)
# #         logits = bc(data)
# #         pred = logits.argmax().view(1)
# #         all_preds.append(pred.cpu())
# #         all_targets.append(target.view(-1).cpu())

# # y_pred = torch.cat(all_preds).numpy().astype(int)
# # y_true = torch.cat(all_targets).numpy().astype(int)

# # # 8. 평가 및 저장
# # test_mask = (y_true == 0) | (y_true == 6)
# # y_pred_mapped = np.where(y_pred == 1, 6, y_pred)
# # acc = (y_pred_mapped[test_mask] == y_true[test_mask]).mean()
# # print(f"accuracy (labels 0/6 only): {acc:.4f}")

# # now = datetime.now().strftime("%Y%m%d_%H%M%S")
# # y_pred_filename = f"y_pred_{now}.csv"
# # np.savetxt(y_pred_filename, y_pred_mapped, fmt="%d")
# # print(f"✅ 저장 완료: {y_pred_filename}")

# # quantum_cnn_qnn_classifier.py
# # quantum_classifier_final.py
# # ✅ QuantumCNNClassifier (0 vs 6 이진 분류 → 전체 예측 결과로 변환 후 저장)

# # [전체 구조 요약]
# # 1. FashionMNIST 0 vs 6만 사용
# # 2. CNN + Quantum Circuit (8 파라미터, 2-qubit)
# # 3. NLLLoss 기반 학습
# # 4. 전체 10,000개 샘플에 대해 0 또는 6으로 예측값 저장
# # quantum_classifier_final.py
# # quantum_submit_exact.py

# import torch
# import torch.nn.functional as F
# from torch import cat
# from torch.nn import Module, Linear, Conv2d, Dropout2d, NLLLoss, BatchNorm2d
# from torch.nn.parameter import Parameter
# from torch.optim import Adam
# from torch.utils.data import DataLoader, Subset

# import pennylane as qml
# import torchvision
# from torchvision import transforms
# from tqdm import tqdm
# import numpy as np
# from datetime import datetime

# # 1. Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_dtype(torch.float64)

# # 2. Data (0 vs 6만 사용)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_ds = torchvision.datasets.FashionMNIST("./", train=True, download=True, transform=transform)
# test_ds = torchvision.datasets.FashionMNIST("./", train=False, download=True, transform=transform)

# train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
# test_mask = (test_ds.targets == 0) | (test_ds.targets == 6)

# train_idx = torch.where(train_mask)[0]
# test_idx = torch.where(test_mask)[0]

# train_ds.targets[train_ds.targets == 6] = 1
# test_ds.targets[test_ds.targets == 6] = 1

# binary_train_ds = Subset(train_ds, train_idx)
# binary_test_ds = Subset(test_ds, test_idx)

# train_loader = DataLoader(binary_train_ds, batch_size=16, shuffle=True)
# test_loader = DataLoader(binary_test_ds, batch_size=16, shuffle=False)

# # 3. Quantum Circuit
# class QuantumCircuit(Module):
#     def __init__(self):
#         super().__init__()
#         self.dev = qml.device("default.qubit", wires=2)
#         self.params = Parameter(torch.rand(8, dtype=torch.float64), requires_grad=True)
#         self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

#         @qml.qnode(self.dev, interface="torch")
#         def circuit(x):
#             qml.AngleEmbedding(x, wires=[0, 1])
#             qml.CNOT(wires=[0, 1])
#             qml.RY(self.params[0], wires=0)
#             qml.RY(self.params[1], wires=1)
#             qml.CNOT(wires=[1, 0])
#             qml.RY(self.params[2], wires=0)
#             qml.RY(self.params[3], wires=1)
#             qml.CNOT(wires=[0, 1])
#             qml.RY(self.params[4], wires=0)
#             qml.RY(self.params[5], wires=1)
#             qml.CNOT(wires=[1, 0])
#             qml.RY(self.params[6], wires=0)
#             qml.RY(self.params[7], wires=1)
#             return qml.expval(self.obs)

#         self.qnode = circuit

#     def forward(self, x):
#         return self.qnode(x)

# # 4. CNN + Quantum Classifier
# class QuantumCNNClassifier(Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2d(1, 8, kernel_size=5)
#         self.bn1 = BatchNorm2d(8)
#         self.conv2 = Conv2d(8, 32, kernel_size=5)
#         self.bn2 = BatchNorm2d(32)
#         self.dropout = Dropout2d(0.3)
#         self.fc1 = Linear(512, 64)
#         self.fc2 = Linear(64, 2)
#         self.qnn = QuantumCircuit()
#         self.final = Linear(1, 1)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2)
#         x = self.dropout(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         q_out = []
#         for i in range(x.size(0)):
#             q_val = self.qnn(x[i])
#             q_out.append(q_val)
#         x = torch.stack(q_out).view(-1, 1)
#         x = self.final(x)
#         return F.log_softmax(cat((x, 1 - x), -1), dim=-1)

# # 5. Init & Check
# model = QuantumCNNClassifier().to(device)
# dummy_input = torch.tensor([0.0, 0.0], dtype=torch.float64)
# specs = qml.specs(model.qnn.qnode)(dummy_input)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# assert specs["num_tape_wires"] <= 8
# assert specs["resources"].depth <= 30
# assert specs["num_trainable_params"] <= 60
# assert total_params <= 50000
# print("✅ 회로 조건 통과")

# # 6. Train
# optimizer = Adam(model.parameters(), lr=0.0005)
# loss_fn = NLLLoss()
# epochs = 20
# model.train()

# for epoch in range(epochs):
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
#     total_loss = 0
#     for data, target in pbar:
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         pbar.set_postfix(loss=f"{loss.item():.4f}")
#     print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_loader):.4f}")

# # 7. Predict only 0/6
# model.eval()
# all_preds = []
# with torch.no_grad():
#     for data, _ in tqdm(test_loader, desc="Inference"):
#         data = data.to(device)
#         logits = model(data)
#         preds = logits.argmax(dim=1)   # 🔥 여기!
#         all_preds.append(preds.cpu())

# y_pred = torch.cat(all_preds).numpy()  # 🔥 전체 2000개 연결
# y_true = test_ds.targets[test_mask].numpy()
# score = (np.where(y_pred == 1, 6, 0) == np.where(y_true == 1, 6, 0)).mean()
# print(f"\n🎯 Score (0 vs 6 only): {score:.4f}")

# # 9. Save (ONLY 0/6)
# y_pred_final = np.zeros(len(test_ds), dtype=int)
# y_pred_final[test_idx] = np.where(y_pred == 1, 6, 0)

# now = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = f"y_pred_{now}.csv"
# np.savetxt(filename, y_pred_final, fmt="%d")
# print(f"📁 저장 완료: {filename}")







import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import NLLLoss, Module, Conv2d, Dropout2d, Linear, BatchNorm2d
from torch.nn.parameter import Parameter
import pennylane as qml
from tqdm import tqdm
from datetime import datetime

# ── Reproducibility / Seed ─────────────────────────
SEED = 6054
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Configuration ─────────────────────────────────
BATCH_SIZE = 512
LR = 5e-4
EPOCHS = 200
PATIENCE = 20
BEST_MODEL_PATH = "best_model.pt"
FULL_MODEL_PATH = "full_model.pt"
SUBMISSION_DIR = "submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ── Device Setup ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ── Data Preparation ──────────────────────────────
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(28, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
train_ds = torchvision.datasets.FashionMNIST(
    "./", train=True, download=True, transform=transform)
test_ds = torchvision.datasets.FashionMNIST(
    "./", train=False, download=True, transform=transform)
# Filter labels 0 and 6 -> map 6 to 1
train_mask = (train_ds.targets == 0) | (train_ds.targets == 6)
test_mask  = (test_ds.targets == 0)  | (test_ds.targets == 6)
train_ds.targets[train_ds.targets == 6] = 1
test_ds.targets[test_ds.targets == 6] = 1
train_idx = torch.where(train_mask)[0]
test_idx  = torch.where(test_mask)[0]
binary_train = Subset(train_ds, train_idx)
binary_test  = Subset(test_ds,  test_idx)
train_loader = DataLoader(binary_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(binary_test,  batch_size=BATCH_SIZE, shuffle=False)

# ── Quantum Circuit ──────────────────────────────
class QuantumCircuit(Module):
    def __init__(self):
        super().__init__()
        self.dev = qml.device("default.qubit", wires=2)
        self.params = Parameter(torch.rand(8, dtype=torch.float64), requires_grad=True)
        self.obs = qml.PauliZ(0) @ qml.PauliZ(1)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            for i in range(4):
                qml.RY(self.params[2*i], wires=0)
                qml.RX(self.params[2*i+1], wires=1)
                if i % 2 == 0:
                    qml.CNOT(wires=[0, 1])
                else:
                    qml.CNOT(wires=[1, 0])
            return qml.expval(self.obs)

        self.circuit = circuit

    def forward(self, x):
        return self.circuit(x)

# ── Model Definition ──────────────────────────────
class QuantumCNNClassifier(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 8, kernel_size=5)
        self.bn1   = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 32, kernel_size=5)
        self.bn2   = BatchNorm2d(32)
        self.dropout = Dropout2d(0.3)
        self.fc1 = Linear(512, 64)
        self.fc2 = Linear(64, 2)
        self.qnn = QuantumCircuit()
        self.final = Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Evaluate quantum circuit on CPU then move result to DEVICE
                # Evaluate quantum circuit on CPU then move result to DEVICE
        q_out = []
        for vec in x:
            cpu_vec = vec.detach().cpu()
            q_val = self.qnn(cpu_vec)
            q_out.append(q_val)
        # Stack and move back to DEVICE
        q_out = torch.stack(q_out).view(-1, 1).float().to(device)
        x = self.final(q_out)
        # Combine two logits into shape [batch, 2]
        x_cat = torch.cat([x, 1 - x], dim=1)
        return F.log_softmax(x_cat, dim=-1)

# ── Train & Save ─────────────────────────────────
model = QuantumCNNClassifier().to(device)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
# Learning rate scheduler: decrease LR by 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
loss_fn = NLLLoss()
best_loss = float('inf')
patience_ctr = 0

for epoch in range(1, EPOCHS + 1):
    scheduler.step()  # Step LR scheduler at start of epoch
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), target.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_ctr = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        # Note: skipping full model save due to QNode unpicklable closure
        # torch.save(model, FULL_MODEL_PATH)
        print(f"❏ Saved {BEST_MODEL_PATH} & {FULL_MODEL_PATH}")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping.")
            break

        print(f"Epoch {epoch} average loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

# ── Inference & Submission ────────────────────────
# Perform inference on the full test set (all 10000 samples)
full_test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
all_preds = []
with torch.no_grad():
    for data, _ in tqdm(full_test_loader, desc="Inference"):
        data = data.to(device)
        preds = model(data).argmax(dim=1).cpu().numpy()
        # Map binary preds: 1 -> 6, 0 -> 0
        mapped = np.where(preds == 1, 6, 0)
        all_preds.extend(mapped.tolist())

# Ensure full coverage
assert len(all_preds) == len(test_ds), \
    f"Expected {len(test_ds)} predictions, got {len(all_preds)}"

# Save submission file
filename = f"y_pred_{datetime.now():%Y%m%d_%H%M%S}.csv"
np.savetxt(filename, all_preds, fmt='%d')
print(f"Saved submission: {filename}")





# [Score] 0 vs 6 only: 0.8750 
# 📁 저장 완료: y_pred_20250724_152830.csv