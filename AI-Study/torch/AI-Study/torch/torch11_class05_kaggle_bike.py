import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
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
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count'] 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=SEED,
    # stratify=y,
)

print(x_train.shape, x_test.shape)  # (8708, 10) (2178, 10)
print(y_train.shape, y_test.shape)  # (8708,) (2178,)

# print(np.unique(y_train))
# exit()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
# 다양한 플랫폼 (특히 GPU)에서 타입 충돌을 피하기 위해,
# PyTorch는 target label은 무조건 int64(long)로 고정했습니다.


print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([8708, 10]) torch.Size([8708, 1])
print(type(x_train))                # <class 'torch.Tensor'>

# exit()

# 2. 모델구성                   // 단순 Sequential 모델
# model = nn.Sequential(
#     nn.Linear(10, 64),
#     nn.ReLU(),
#     nn.Linear(64, 512),
#     nn.ReLU(),
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#         # nn.SiLU(),
#     nn.ReLU(),
#     nn.Linear(32, 1),
#     # nn.Softmax(),
# ).to(DEVICE)

class Model(nn.Module):   # 모델이 복잡해지면 이렇게 함수화 하는게 용이함
    def __init__(self, input_dim, output_dim):           # 정의만 한 것
        super().__init__()              # 두개 다 같음
        # super(Model, self).__init__() # nn.Module에 있는 Model과 self 다 쓰겠다는 뜻.
        ### 모델에 대한 정의부분을 구현 ###
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout =nn.Dropout(0.2)

    def forward(self, x):                # 정의 구현하는 것. forward method는 nn.Module 안에 있는 놈
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear6(x)
        
        return x
        
model = Model(10, 1).to(DEVICE)



# 3. 컴파일, 훈련
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss() # Sparse Categorical Entropy. 원핫과 소프트맥스가 포함.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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


# Final Loss: 182.18118286132812
# r2: 0.9946381717658409







exit()
#######################################################
y_predict = model(x_test).cpu() # acc 빼기위함
y_predict_cls = (y_predict >= 0.5).int().numpy()
y_true = y_test.cpu().numpy().astype(int)

acc = accuracy_score(y_true, y_predict_cls)
print('acc:', acc)
# Final Loss: 0.8519402146339417
# acc: 0.9590643274853801

