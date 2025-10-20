from torchvision.datasets import CIFAR100
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import TensorDataset, DataLoader

####################### 랜덤 고정 #########################
SEED = 707
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
##########################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

path = './_data/torch/'

# 1) CIFAR-100 로드
train_dataset = CIFAR100(path, train=True,  download=True)
test_dataset  = CIFAR100(path, train=False, download=True)

# 2) NumPy → torch.Tensor, 0~1 정규화, flatten
x_train = torch.tensor(train_dataset.data, dtype=torch.float32) / 255.0  # (50000,32,32,3)
x_test  = torch.tensor(test_dataset.data,  dtype=torch.float32) / 255.0  # (10000,32,32,3)
y_train = torch.tensor(train_dataset.targets, dtype=torch.long)         # (50000,)
y_test  = torch.tensor(test_dataset.targets,  dtype=torch.long)         # (10000,)

# 3) 3*32*32=3072 차원으로 일렬로 펴기
x_train = x_train.reshape(-1, 3*32*32)  # (50000,3072)
x_test  = x_test.reshape(-1, 3*32*32)   # (10000,3072)

print(x_train.shape, y_train.shape)  # torch.Size([50000,3072]) torch.Size([50000])
print(x_test.shape,  y_test.shape)   # torch.Size([10000,3072]) torch.Size([10000])

# 4) DataLoader
train_set   = TensorDataset(x_train, y_train)
test_set    = TensorDataset(x_test,  y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# 5) DNN 정의 (출력 노드만 100개)
class DNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DNN, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128), nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU()
        )
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.output_layer(x)

model = DNN(num_features=3*32*32, num_classes=100).to(DEVICE)

# 6) 손실함수·옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 7) train 함수
def train(model, criterion, optimizer, loader):
    model.train()
    total_loss, total_acc = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y_batch).float().mean()
        total_loss += loss.item()
        total_acc  += acc.item()
    return total_loss/len(loader), total_acc/len(loader)

# 8) evaluate 함수
def evaluate(model, criterion, loader):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y_batch).float().mean()
            total_loss += loss.item()
            total_acc  += acc.item()
    return total_loss/len(loader), total_acc/len(loader)

# 9) 학습 루프
EPOCH = 20
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    print(f'epoch:{epoch:02d}  loss={loss:.4f}  acc={acc:.3f}')

print('-'*80)

# 10) 최종 평가
final_loss, final_acc = evaluate(model, criterion, test_loader)
print(f'Final Loss: {final_loss:.4f}  Final ACC: {final_acc:.3f}')


# Final Loss: 3.6710  Final ACC: 0.140