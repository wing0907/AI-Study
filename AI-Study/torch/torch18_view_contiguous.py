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
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

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
            input_size=1,           # 입력 특성(feature) 차원 // feature 개수 // 텐서플로우에서는 input_dim
            hidden_size=32,         # 은닉 상태(hidden state) 차원 // 출력(output) 노드 개수와 동일 (RNN의 내부 표현 차원) // 텐서플로우에서는 unit
            # num_layers=1,           # RNN 층 수 // 예: num_layers=2이면 첫 번째 RNN의 출력이 두 번째 RNN의 입력으로 들어감. // 디폴트가 1, 1이면 단층 RNN, 2 이상이면 Stacked RNN (깊은 RNN) 
            batch_first=True,       # 입력 텐서 차원 배치 우선 여부 // 입력 텐서의 차원 순서를 (배치, 시퀀스 길이, 특성) 로 받겠다는 의미 // 기본값은 False → (시퀀스 길이, 배치, 특성) 순서 // 보통 직관적으로 다루기 위해 batch_first=True를 많이 씀.
        )                           
        # self.rnn_layer2 = nn.RNN(32, 32, batch_first=True) # 위에 주석하고 이렇게 처리해도 돌아감.
        self.fc1 = nn.Linear(32, 16)  # fully connected layer = 전체가 전부 연결된 구조. Dense 밀집 
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn_layer1(x)   # RNN 통과 → (batch, seq, hidden) // x, _ = self.cell (위에도 self.cell 로 정의하면 됨)
        x = self.relu(x)            # x, _ = 아웃풋을 명시하지 않겠다

        
        ##### 1. reshape : 잘돌아감
        # x = x.reshape(-1, 3*32)     # Flatten 해도 무방함 // Flatten (seq_len*hidden)

        ##### 2. view : 에러
        # x = x.view(-1, 3*32)      # 연속적 메모리 데이터에만 해당
                                    # 연속적 메모리 데이터와 연속적이지 않은 메모리 데이터의 차이 :
        ##### 3. contiguous 를 이용해서 연속적인 메모리 라고 인식시킨다.
        x = x.contiguous()
        x = x.view(-1, 3*32)


        # x = x[:, -1, :]             # Torch의 슬라이싱  // 2,817 의 파라미터 개수가 1000 몇개로 줄어들지만 성능은 비슷함
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