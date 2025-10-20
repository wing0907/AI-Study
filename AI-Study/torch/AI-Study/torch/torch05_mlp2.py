import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()     # 대문자는 개발자들끼리 상수로 인정한다. 즉 고정값
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,  '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
                [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.transpose(x)

print(x.dtype)  # float64
print(x.shape, y.shape)     # (10, 3) (10,)

# x = torch.FloatTensor(x).to(DEVICE) 
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

###### 아래가 권장 문법 ######
x = torch.tensor(x, dtype=torch.float).to(DEVICE) 
y = torch.tensor(y, dtype=torch.float).unsqueeze(1).to(DEVICE) 


x_mean = torch.mean(x)
x_std = torch.std(x)
x = (x - x_mean) / x_std
print('scaled x:', x)
# scaled x: tensor([[-0.9502, -0.9502,  1.7485],
#         [-0.6128, -0.9164,  1.4112],
#         [-0.2755, -0.8827,  1.0739],
#         [ 0.0618, -0.8490,  0.7365],
#         [ 0.3992, -0.8152,  0.3992],
#         [ 0.7365, -0.7815,  0.0618],
#         [ 1.0739, -0.7478, -0.2755],
#         [ 1.4112, -0.7140, -0.6128],
#         [ 1.7485, -0.6803, -0.9502],
#         [ 2.0859, -0.6466, -1.2875]], device='cuda:0', dtype=torch.float64)


model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.08)

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

x_pred = (torch.Tensor([[[11, 2.0, -1]]]).to(DEVICE) - x_mean) / x_std

result = model(x_pred)
print('11의 예측값:', result.item())


