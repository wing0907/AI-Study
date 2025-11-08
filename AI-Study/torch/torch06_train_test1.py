import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()     # 대문자는 개발자들끼리 상수로 인정한다. 즉 고정값
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,  '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10,11])
y_test = np.array([8,9,10,11])

x_pred = np.array([12,13,14])

x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)




print(x_train.dtype)  # torch.float32
print(x_train.size(), y_train.size())     # torch.Size([7, 1]) torch.Size([7, 1])



# x = torch.FloatTensor(x).to(DEVICE) 
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

###### 아래가 권장 문법 ######
x = torch.tensor(x_train, dtype=torch.float).to(DEVICE) 
y = torch.tensor(y_train, dtype=torch.float).to(DEVICE) 



x_mean = torch.mean(x_train)
x_std = torch.std(x_train)
x = (x - x_mean) / x_std
print('scaled x:', x)

# scaled x: tensor([[-1.3887],
#         [-0.9258],
#         [-0.4629],
#         [ 0.0000],
#         [ 0.4629],
#         [ 0.9258],
#         [ 1.3887]], device='cuda:0')

model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 700
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
print('----------------------------------------------------')    


def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print('Final Loss:', loss2)

x_pred = (x_pred - x_mean) / x_std

result = model(x_pred)
print('예측값:\n', result.tolist())
# print('예측값:\n', result.detach())           # 2개 이상 툴력할때. device='cuda:0' 까지 같이 나옴
# print('예측값:\n', result.detach().cpu().numpy())  # numpy 연산은 gpu가 없음. 그래서 cpu 씀

# Final Loss: 1.0759647431921385e-13
# 예측값:
#  [[11.999999046325684], [13.0], [14.0]]
