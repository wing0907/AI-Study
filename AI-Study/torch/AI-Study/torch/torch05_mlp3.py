import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()     # 대문자는 개발자들끼리 상수로 인정한다. 즉 고정값
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,  '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])

x = x.T      #  (3, 10)
y = y.T     # (2, 10)
  

print(x.dtype)  # int32
print(x.shape, y.shape)     # (10, 3) (10, 2)


# x = torch.FloatTensor(x).to(DEVICE) 
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

###### 아래가 권장 문법 ######
x = torch.tensor(x, dtype=torch.float).to(DEVICE) 
y = torch.tensor(y, dtype=torch.float).to(DEVICE) 


x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x - x_mean) / x_std
print('scaled x:', x)

# scaled x: tensor([[-0.8551, -0.6264,  1.3344],
#         [-0.8442, -0.6155,  1.3453],
#         [-0.8333, -0.6046,  1.3562],
#         [-0.8224, -0.5937,  1.3671],
#         [-0.8116, -0.5828,  1.3780],
#         [-0.8007, -0.5719,  1.3889],
#         [-0.7898, -0.5610,  1.3998],
#         [-0.7789, -0.5501,  1.4107],
#         [-0.7680, -0.5392,  1.4216],
#         [-0.7571, -0.5283,  1.4325]], device='cuda:0')



model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 2),
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

epochs = 500
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

x_pred = (torch.Tensor([[10, 31, 211], [11, 32, 212]]).to(DEVICE)- x_mean) / x_std

result = model(x_pred)
# print('예측값:\n', result.tolist())
# print('예측값:\n', result.detach())           # 2개 이상 툴력할때. device='cuda:0' 까지 같이 나옴
print('예측값:\n', result.detach().cpu().numpy())  

# Final Loss: 0.020618392154574394
# 예측값:
#  [[10.903208    0.09915483]
#  [11.909363   -0.9104434 ]]
