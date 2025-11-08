from torchvision.datasets import MNIST
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
from torch.utils.data import TensorDataset # x, y 합치기
from torch.utils.data import DataLoader    # batch 정의!!!

# import warnings
# warnings.filterwarnings('ignore')


####################### 랜덤 고정 #########################
SEED = 707
random.seed(SEED)               # 파이썬 랜덤 고정
np.random.seed(SEED)            # 넘파이 랜덤 고정
torch.manual_seed(SEED)         # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)    # 토치 쿠다 랜덤 고정
##########################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', )


path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True) #.to(DEVICE)
test_dataset = MNIST(path, train=False, download=True)  #.to(DEVICE)

print(train_dataset)
# Dataset MNIST
#     Number of datapoints: 60000
#     Root location: ./_data/torch/
#     Split: Train
print(type(train_dataset))      # <class 'torchvision.datasets.mnist.MNIST'>
print(train_dataset[59999])     # (<PIL.Image.Image image mode=L size=28x28 at 0x2AB6A30BF70>, 8)

x_train, y_train = train_dataset.data/255., train_dataset.targets
x_test, y_test = test_dataset.data/255., test_dataset.targets

print(x_train)
# tensor([[[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]],

#         [[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]],

#         [[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]],

#         ...,

#         [[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]],

#         [[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]],

#         [[0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          ...,
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0],
#          [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8)
print(y_train)
# tensor([5, 0, 4,  ..., 5, 6, 8])

print(x_train.shape, y_train.size())  # torch.Size([60000, 28, 28]) torch.Size([60000])

print(np.min(x_train.numpy()), np.max(x_train.numpy()))     # 0 255  /255. 해서 0.0 1.0 이렇게 됨.

x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)  # 토치에서는 .view 를 더 많이 씀
print(x_train.shape, x_test.size())  # torch.Size([60000, 784]) torch.Size([10000, 784])


train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 2. 모델 
class DNN(nn.Module):
    def __init__(self, num_features):
        # super().__init__()      # 두가지 중에 하나 쓰면 됨.
        super(DNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = DNN(784).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1e-4) # 0.0001

def train(model, criterion, optimizer, loader):
    # model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step() # w = w - lr * loss를 weight로 미분한 값
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

EPOCH = 20
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    print(f'epoch: {epoch}, loss: {loss:.4f}, acc: {acc:.3f}')

print('--------------------------------------------------------------------------------------')


def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
                       
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc
        return epoch_loss / len(loader), epoch_acc / len(loader)

EPOCH = 10
for epoch in range(1, EPOCH+1):
    loss, acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = evaluate(model, criterion, test_loader)  # 나중에는 test_loader 대신 val_loader 또는 eval_loader 라고 데이터를 따로 분리해서 사용하면 가독성 굿.
    print(f'epoch: {epoch}, loss: {loss:.4f}, acc: {acc:.3f},\
          val_loss: {val_loss:.4f}, val_acc:{val_acc:.3f}'
          )
    
# 4. 평가, 예측
loss, acc = evaluate(model, criterion, test_loader)
print('-----------------------------------------------------------------------')
print('Final Loss:', loss)
print('Final ACC:', acc.detach().cpu().numpy())

# Final Loss: 0.32088401402456884
# Final ACC: 0.91044325

# dropout 적용 후
# Final Loss: 0.24486671890260076
# Final ACC: 0.92991215