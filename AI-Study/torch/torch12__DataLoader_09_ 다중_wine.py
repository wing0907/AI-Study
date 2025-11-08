import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
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

datasets = load_wine()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=SEED,
    stratify=y,
)

# print(x_train.shape, x_test.shape)     # (124, 13) (54, 13)
# print(y_train.shape, y_test.shape)     # (124,) (54,)
# print(np.unique(y_train))              # [0 1 2]  = int64 (분류 // 다중) // 이진은 상진+우진 
# exit()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

y_train = torch.tensor(y_train, dtype=torch.int64).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)

print(x_train.dtype)                         # torch.float32
print(x_train.shape, y_train.shape)          # torch.Size([105, 4]) torch.Size([105, 1])
print(type(x_train))                         # <class 'torch.Tensor'>

# exit()
############################# torch 데이터셋 만들기 ##############################
from torch.utils.data import TensorDataset # x, y 합치기
from torch.utils.data import DataLoader    # batch 정의!!!

########## 1. x와 y를 합쳤다.
train_set = TensorDataset(x_train, y_train)     # tuple 형태로
test_set = TensorDataset(x_test, y_test) 
print(train_set)     # <torch.utils.data.dataset.TensorDataset object at 0x0000023C75515AB0>
print(type(train_set)) # <class 'torch.utils.data.dataset.TensorDataset'>
print(len(train_set)) # 124
print(train_set[0])
# (tensor([-0.3398,  1.2653,  0.0959,  0.9952,  0.0315,  0.8935,  0.5461,  0.5662,
#          0.6678, -1.0497,  0.9096,  0.7456, -0.8582], device='cuda:0'), tensor(1, device='cuda:0'))
print(train_set[0][0])      # 첫번째 x
# tensor([-0.3398,  1.2653,  0.0959,  0.9952,  0.0315,  0.8935,  0.5461,  0.5662,
#          0.6678, -1.0497,  0.9096,  0.7456, -0.8582], device='cuda:0')
print(train_set[0][1])      # 첫번째 y
# tensor(1, device='cuda:0')

# exit()
####### 2. batch를 정의한다
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
print(len(train_loader)) # 7
print(train_loader)      # <torch.utils.data.dataloader.DataLoader object at 0x000001940B876620>
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
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, 16)
        self.linear6 = nn.Linear(16, output_dim)
        # self.softmax = nn.Softmax()
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
        # x = self.softmax(x)

        return x

model = Model(13, 3).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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


# Final Loss: 0.19988749003660544
# acc: 0.9814814814814815






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