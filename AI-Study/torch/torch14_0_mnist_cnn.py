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

# 1. 데이터
import torchvision.transforms as tr
transf = tr.Compose([tr.Resize(56), tr.ToTensor()])
# to.Tensor = 토치텐서바꾸기 + minmaxScaler


path = './_data/torch/'
train_dataset = MNIST(path, train=True, download=True, transform=transf) #.to(DEVICE)
test_dataset = MNIST(path, train=False, download=True, transform=transf)  #.to(DEVICE)
print(len(train_dataset))    # 60000
# print(train_dataset[0][0])
print(train_dataset[0][1])   # 5

img_tensor, label = train_dataset[0]  # 튜플 데이터는 이런식으로 뺄 수 있음
print(label)                 # 5
print(img_tensor.shape)      # torch.Size([1, 56, 56])    # uint8 H×W×C → FloatTensor C×H×W in [0,1] CNN은 이렇게 바껴야함
print(img_tensor.min(), img_tensor.max())  # tensor(0.) tensor(0.9922) => 전처리가 되어 있다는 뜻



# x_train, y_train = train_dataset.data/255., train_dataset.targets
# x_test, y_test = test_dataset.data/255., test_dataset.targets

# print(np.min(x_train.numpy()), np.max(x_train.numpy()))     # 0 255  /255. 해서 0.0 1.0 이렇게 됨.

# x_train, x_test = x_train.view(-1, 28*28), x_test.reshape(-1, 784)  # 토치에서는 .view 를 더 많이 씀
# print(x_train.shape, x_test.size())  # torch.Size([60000, 784]) torch.Size([10000, 784])

# train_set = TensorDataset(x_train, y_train)
# test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(len(train_loader)) # 1875 = 60000 / 32

# exit()

# 2. 모델 
class CNN(nn.Module):
    def __init__(self, num_features):
        # super().__init__()      # 두가지 중에 하나 쓰면 됨.
        super(CNN, self).__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3,3), stride=1),   # (1,56,56) -> (64,54,54)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2), # default => stride=2   # (n, 64, 27, 27)
            nn.Dropout(0.2),
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), stride=1),     # (n, 32, 25, 25)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),        # (n, 32, 12, 12)
            nn.Dropout(0.2),
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1),  # (n, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),        # (n, 16, 5, 5)
            nn.Dropout(0.2),
        )
        self.hidden_layer4 = nn.Sequential(     # Flatten에서 받아라
            nn.Linear(16*5*5, 64),
            nn.ReLU(),
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
        x = x.view(x.shape[0], -1)  # Flatten 실행
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x
    
model = CNN(1).to(DEVICE)  # torch에서는 channel 만 input으로 넣어줌. 나머지는 알아서 맞춰줌

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

# Final Loss: 0.07560158569065217
# Final ACC: 0.975639