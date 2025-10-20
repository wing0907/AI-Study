import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import random

####################### 랜덤 고정 #########################
SEED = 337
random.seed(SEED)               # 파이썬 랜덤 고정
np.random.seed(SEED)            # 넘파이 랜덤 고정
torch.manual_seed(SEED)         # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)    # 토치 쿠다 랜덤 고정
##########################################################

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', )

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=222,
    # stratify=y,
)

print(x_train.shape, x_test.shape)  # (18576, 8) (2064, 8)
print(y_train.shape, y_test.shape)  # (18576,) (2064,)

# print(np.unique(y_train))
# exit()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)



print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([18576, 8]) torch.Size([18576, 1])
print(type(x_train))                # <class 'torch.Tensor'>

# exit()

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.SiLU(),
    nn.Linear(16, 1),
    # nn.Softmax(),
).to(DEVICE)

# 3. 컴파일, 훈련
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss() # Sparse Categorical Entropy. 원핫과 소프트맥스가 포함.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss =criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()


epochs = 600
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs: {}, loss: {}'.format(epoch, loss))  # verbose
print('----------------------------------------------------')


# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()
final_loss = evaluate(model, criterion, x_test, y_test)
print('Final Loss:', final_loss)


######################################################
y_predict = model(x_test)
# print(type(y_predict))

# y_predict = y_predict.detach()
# print(y_predict)
# y_predict = y_predict.cpu()
# print(y_predict)
# print(type(y_predict))
# y_predict = y_predict.numpy()
# print(type(y_predict))
y_predict = np.round(y_predict.detach().cpu().numpy())



y_test = y_test.detach().cpu().numpy()

r2 = r2_score(y_test, y_predict)
print('r2:', r2)


# Final Loss: 0.2467401772737503
# r2: 0.7474910724561623







exit()
#######################################################
y_predict = model(x_test).cpu() # acc 빼기위함
y_predict_cls = (y_predict >= 0.5).int().numpy()
y_true = y_test.cpu().numpy().astype(int)

acc = accuracy_score(y_true, y_predict_cls)
print('acc:', acc)
# Final Loss: 0.8519402146339417
# acc: 0.9590643274853801

