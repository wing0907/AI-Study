import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
from sklearn.preprocessing import LabelEncoder

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

path = 'C:\Study25\_data\kaggle\otto\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']


# 2. 라벨 인코딩
le = LabelEncoder()
y_enc = le.fit_transform(y)   # now 0~8
print(np.unique(y, return_counts=True))

# 3. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y_enc, train_size=0.75, random_state=SEED,
    stratify=y_enc
)

print(x_train.shape, x_test.shape)  # (46408, 93) (15470, 93)
print(np.unique(y_train))           # [0 1 2 3 4 5 6 7 8]
# exit()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.int64).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)

print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([46408, 93]) torch.Size([46408])
print(type(x_train))                # <class 'torch.Tensor'>

############################# torch 데이터셋 만들기 ##############################
from torch.utils.data import TensorDataset # x, y 합치기
from torch.utils.data import DataLoader    # batch 정의!!!

########## 1. x와 y를 합쳤다.
train_set = TensorDataset(x_train, y_train)     # tuple 형태로
test_set = TensorDataset(x_test, y_test) 
print(train_set)     # <torch.utils.data.dataset.TensorDataset object at 0x000001D0135F8D60>
print(type(train_set)) # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set)) # 46408
print(train_set[0])
# (tensor([-0.2560, -0.2103,  0.3778, -0.2764, -0.1591, -0.1217, -0.1911, -0.2973,
#         -0.2917, -0.2436, -0.4114, -0.2454, -0.2375, -0.5360, -0.3330, -0.1842,
#         -0.2485, -0.4346, -0.1195, -0.3281, -0.2940,  0.1899, -0.1793, -0.3695,
#         -0.2289,  0.2599, -0.2464, -0.2886, -0.1428, -0.0911, -0.1653, -0.4265,
#         -0.5304, -0.2818, -0.2125, -0.2775, -0.3529, -0.3604, -0.1468, -0.5003,
#         -0.2750, -0.3519, -0.2625, -0.4162, -0.0924, -0.2733, -0.1710, -0.2304,
#         -0.2514, -0.1889, -0.1022, -0.1982, -0.2203, -0.2592, -0.3214, -0.1563,
#          5.1908, -0.1748, -0.1497, -0.3410, -0.2347, -0.0694, -0.1772, -0.3680,
#         -0.2863,  0.3208, -0.5813,  1.4762, -0.1983, -0.4577, -0.2788, -0.3555,
#         -0.1110, -0.1562, -0.2282, -0.2592, -0.1360, -0.1335, -0.2330, -0.3085,
#         -0.1697, -0.2233, -0.2069, -0.0615, -0.2810, -0.4220, -0.2526, -0.4142,
#         -0.3034, -0.1756, -0.1328, -0.3875, -0.1144], device='cuda:0'), tensor(5, device='cuda:0'))
print(train_set[0][0])      # 첫번째 x
# tensor([-0.2560, -0.2103,  0.3778, -0.2764, -0.1591, -0.1217, -0.1911, -0.2973,
#         -0.2917, -0.2436, -0.4114, -0.2454, -0.2375, -0.5360, -0.3330, -0.1842,
#         -0.2485, -0.4346, -0.1195, -0.3281, -0.2940,  0.1899, -0.1793, -0.3695,
#         -0.2289,  0.2599, -0.2464, -0.2886, -0.1428, -0.0911, -0.1653, -0.4265,
#         -0.5304, -0.2818, -0.2125, -0.2775, -0.3529, -0.3604, -0.1468, -0.5003,
#         -0.2750, -0.3519, -0.2625, -0.4162, -0.0924, -0.2733, -0.1710, -0.2304,
#         -0.2514, -0.1889, -0.1022, -0.1982, -0.2203, -0.2592, -0.3214, -0.1563,
#          5.1908, -0.1748, -0.1497, -0.3410, -0.2347, -0.0694, -0.1772, -0.3680,
#         -0.2863,  0.3208, -0.5813,  1.4762, -0.1983, -0.4577, -0.2788, -0.3555,
#         -0.1110, -0.1562, -0.2282, -0.2592, -0.1360, -0.1335, -0.2330, -0.3085,
#         -0.1697, -0.2233, -0.2069, -0.0615, -0.2810, -0.4220, -0.2526, -0.4142,
#         -0.3034, -0.1756, -0.1328, -0.3875, -0.1144], device='cuda:0')
print(train_set[0][1])      # 첫번째 y
# tensor(5, device='cuda:0')

# exit()

####### 2. batch를 정의한다
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
print(len(train_loader)) # 465
print(train_loader)      # <torch.utils.data.dataloader.DataLoader object at 0x0000023CA0CE5C90>
# print(train_loader[0])      # 에러
# print(train_loader[0][0])   # 에러


# exit()
print('---------------------------------------------------------------------------------------------')
############# 이터레이터 데이터 확인하기 ###############
# 1. for문으로 확인
# for aaa in train_loader:
#     print(aaa)
#     break               # 첫번째 배치 출력
print('---------------------------------------------------------------------------------------------')
for x_batch, y_batch in train_loader:
    print(x_batch)
    print(y_batch)
    break               # 첫번째 배치 출력


print('---------------------------------------------------------------------------------------------')
# 2. next() 사용
bbb = iter(train_loader)
# aaa = bbb.next()        # 파이썬 버전업 후 .next()는 없어져서 안써 이놈아!!!!!! 이색꺄!!!! 안쓴다!!!!
aaa = next(bbb)
print(aaa)



# exit()

# 2. 모델구성
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.SiLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear6(x)

        return x

model = Model(93, 9).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:   # epoch 단위훈련에서 batch 단위로 훈련이 된다
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss =criterion(hypothesis, y_batch)

        loss.backward()
        optimizer.step()
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()

    return total_loss / len(loader)

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs: {}, loss: {}'.format(epoch, loss))  # verbose
print('----------------------------------------------------')

# exit()

# 4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            
            y_pred = model(x_batch)
            loss2 = criterion(y_pred, y_batch)
            total_loss += loss2.item()
            
            # 예측과 정답을 저장
            all_preds.append(y_pred.detach().cpu())
            all_targets.append(y_batch.detach().cpu())
    # 전체를 합치기
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    return total_loss / len(loader), all_preds, all_targets

# 호출
final_loss , y_predict, y_true = evaluate(model, criterion, test_loader)
print('Final Loss:', final_loss)

# 후처리
y_predict = np.argmax(y_predict, axis=1)
acc = accuracy_score(y_true, y_predict)
print('acc:', acc)

# Final Loss: 1.761389607767905
# acc: 0.3608274078862314


##################################################
# print('=======================================')
# y_pred_labels = []
# y_test_labels = []
# #4. 평가, 예측
# def evaluate(model, cri, testloader):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for x_batch, y_batch in testloader:
#             y_pred = model(x_batch)
#             loss2 = cri(y_pred, y_batch)
#             total_loss += loss2.item()
#             y_pred_labels.append(y_pred.cpu())
#             y_test_labels.append(y_batch.cpu())
#     y_pred_final = torch.cat(y_pred_labels, dim=0).squeeze().numpy()
#     y_test_final = torch.cat(y_test_labels, dim=0).squeeze().numpy()
#     return total_loss/len(testloader),y_pred_final,y_test_final

# loss2,y_pred_final,y_test_final = evaluate(model, criterion, test_loader)

# y_pred_round = (y_pred_final>0.5).astype(int) #np.round() 사용 예: 이진 분류 (y_predict > 0.5).astype(int)
# # y_pred_labels = np.argmax(y_pred_labels, axis=1) #np.argmax(axis=1) 사용 예: 다중 클래스 분류

# acc = accuracy_score(y_test_final, y_pred_round)