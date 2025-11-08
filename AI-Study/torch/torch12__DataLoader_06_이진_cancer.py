import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
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

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=SEED,
    stratify=y,
)

# print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
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
print(x_train.shape, y_train.shape) # torch.Size([398, 30]) torch.Size([398, 1])
print(type(x_train))                # <class 'torch.Tensor'>

############################# torch 데이터셋 만들기 ##############################
from torch.utils.data import TensorDataset # x, y 합치기
from torch.utils.data import DataLoader    # batch 정의!!!

########## 1. x와 y를 합쳤다.
train_set = TensorDataset(x_train, y_train)     # tuple 형태로
test_set = TensorDataset(x_test, y_test) 
print(train_set)     # <torch.utils.data.dataset.TensorDataset object at 0x000001D0135F8D60>
print(type(train_set)) # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set)) # 398
print(train_set[0])
# (tensor([-0.6010, -1.3487, -0.5968, -0.6086,  0.9639, -0.2585, -0.6335, -0.5410,
#         -0.5630,  0.4602, -0.5467, -1.3047, -0.5602, -0.4677,  0.1369, -0.7857,
#         -0.4264, -0.6009, -0.0898, -0.5675, -0.6025, -1.6066, -0.6214, -0.5974,
#          0.9008, -0.5698, -0.5253, -0.6196, -0.1714, -0.1629], device='cuda:0'), tensor([1.], device='cuda:0'))
print(train_set[0][0])      # 첫번째 x
# tensor([-0.6010, -1.3487, -0.5968, -0.6086,  0.9639, -0.2585, -0.6335, -0.5410,
#         -0.5630,  0.4602, -0.5467, -1.3047, -0.5602, -0.4677,  0.1369, -0.7857,
#         -0.4264, -0.6009, -0.0898, -0.5675, -0.6025, -1.6066, -0.6214, -0.5974,
#          0.9008, -0.5698, -0.5253, -0.6196, -0.1714, -0.1629], device='cuda:0')
print(train_set[0][1])      # 첫번째 y
# tensor([1.], device='cuda:0')

####### 2. batch를 정의한다
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
print(len(train_loader)) # 13
print(train_loader)      # <torch.utils.data.dataloader.DataLoader object at 0x0000023CA0CE5C90>
# print(train_loader[0])      # 에러
# print(train_loader[0][0])   # 에러

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
        self.sigmoid = nn.Sigmoid()
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
        x = self.sigmoid(x)

        return x

model = Model(30, 1).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

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

epochs = 200
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    print('epochs: {}, loss: {}'.format(epoch, loss))  # verbose
print('----------------------------------------------------')

# exit()

# 4. 평가, 예측
def evaluate(model, criterion, loader): # 통배치해도 무방함
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # 기울기 계산이 될 수 있으니 no_grad를 사용함
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
y_predict = np.round(y_predict)
acc = accuracy_score(y_true, y_predict)
print('acc:', acc)

# Final Loss: 1.498999496921897
# acc: 0.9649122807017544

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