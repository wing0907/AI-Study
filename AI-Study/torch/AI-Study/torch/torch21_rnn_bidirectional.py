from torchvision.datasets import MNIST
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader

# import warnings
# warnings.filterwarnings('ignore')


####################### 랜덤 고정 #########################
SEED = 316
random.seed(SEED)               # 파이썬 랜덤 고정
np.random.seed(SEED)            # 넘파이 랜덤 고정
torch.manual_seed(SEED)         # 토치 랜덤 고정
torch.cuda.manual_seed(SEED)    # 토치 쿠다 랜덤 고정
##########################################################

# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('torch:', torch.__version__, '사용 device:', DEVICE)

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch:', torch.__version__, '사용 device:', DEVICE)

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열 데이터가 될 수 있다. (온도, 월급 등 시간순서대로 이어져있다고 볼 수 있음)
# 시계열 데이터가 3차원 이지만, 2차원 데이터로 받을 수 있다. 그럴땐 timesteps와 features를 나누는 방식으로 reshape 해줘야 한다.
# 시계열 데이터는 y 값을 주지 않는다. 실무에선 datasets 처럼 데이터를 받게 된다. x와 y값을 n단위로 짤라서 분류하는건 나의 몫. 함수로 지정할 수 있음.
# 1,2,3 다음은 4 / 2,3,4 다음은 5로 학습시키는 것. 8, 9, 10 다음은 데이터가 없음으로 9까지 나누고 8,9,10 다음 11을 예측하는 설계를 함.

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],])        # (7, 3)
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3), (7,)
x = x.reshape(x.shape[0], x.shape[1], 1)    # RNN 입력: (배치, 시퀀스길이, 특성수)
print(x.shape)  # (7, 3, 1)


# ---------------------------------------------------------------------#
# # x = torch.FloatTensor(x).to(DEVICE)
# x = torch.Tensor(x, dtype=torch.float32).to(DEVICE)
# y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
# print(x.shape, y.size())    # torch.Size([7, 3, 1]) torch.Size([7])

# train_set = TensorDataset(x, y)
# train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# aaa = iter(train_loader)
# bbb =next(aaa)
# print(bbb)

# -------------------------------
# 1. 데이터 분할
# -------------------------------
# x_train, x_test, y_train, y_test = train_test_split(
#     x.cpu().numpy(), 
#     y.cpu().numpy(),
#     test_size=0.2,  # 20%를 검증용
#     random_state=SEED,
#     shuffle=True
# )

# -------------------------------
# 1. 학습/테스트 분할
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,  # 20% 검증
    random_state=SEED,
    shuffle=False   # 시계열은 순서 유지
)

# -------------------------------
# 2. 텐서 변환
# -------------------------------
x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# -------------------------------
# 3. TensorDataset + DataLoader
# -------------------------------
# train_set = TensorDataset(x_train, y_train)
# test_set = TensorDataset(x_test, y_test)

# train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

# -------------------------------
# 3. DataLoader
# -------------------------------
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=2, shuffle=False)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False)



# 2. 모델
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_layer1 = nn.RNN(
            input_size=1,           
            hidden_size=32,         
            # num_layers=1,           
            batch_first=True,       
            bidirectional=True,     # default = False
        )                           
            # (N, 3, 32) bidirectional 하니깐 2배가 됨         
              
        # self.rnn_layer2 = nn.RNN(32, 32, batch_first=True) 

        # self.fc1 = nn.Linear(3*32*2, 16)  # 그래서 3*32*2 가 됨  ******
        self.fc1 = nn.Linear(32*2, 16)      # 위 또는 밑에거 사용하면 됨
        
        
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn_layer1(x)   # RNN 통과 → (batch, seq, hidden) // x, _ = self.cell (위에도 self.cell 로 정의하면 됨)
        x = self.relu(x)            # x, _ = 아웃풋을 명시하지 않겠다
        
        
        
        # x = x.reshape(-1, 3*32*2)     # ******   
        x = x[:, -1, :]                 # 아래거 사용하면 요놈 쓰기

        x = self.fc1(x)         
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)             # 최종 출력 (batch,1)
        return x
 
model = RNN().to(DEVICE)

from torchsummary import summary
from torchinfo import summary
# summary(model, (3, 1))
summary(model, (2, 3, 1))


# 3. 컴파일, 훈련
criterion = nn.MSELoss()  # 회귀 문제 → MSELoss로 변경
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    model.train()
    epoch_loss = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        y_batch = y_batch.view(-1, 1)  # (batch,1)로 reshape
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_batch = y_batch.view(-1, 1)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            epoch_loss += loss.item()
            
            preds.append(hypothesis.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    # RMSE 계산
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()
    rmse = np.sqrt(np.mean((preds - trues)**2))
    return epoch_loss / len(loader), rmse, preds, trues

EPOCH = 50
for epoch in range(1, EPOCH+1):
    train_loss = train(model, criterion, optimizer, train_loader)
    val_loss, val_rmse, _, _ = evaluate(model, criterion, test_loader)
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.3f}')


# 4. 최종 평가
loss, rmse, preds, trues = evaluate(model, criterion, test_loader)
print('-'*100)
print('Final Test Loss:', loss)
print('Final Test RMSE:', rmse)
print('실제값:', trues)
print('예측값:', preds)

# Final Test Loss: 74.54061889648438
# Final Test RMSE: 8.633691
# 실제값: [ 9. 10.]
# 예측값: [0.8614856  0.89803225]