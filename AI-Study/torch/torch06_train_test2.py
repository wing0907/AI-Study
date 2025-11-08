import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()     # 대문자는 개발자들끼리 상수로 인정한다. 즉 고정값
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,  '사용 device:', DEVICE)
# torch: 2.7.1+cu118 사용 device: cuda

# 1. 데이터
x = np.array(range(100))
y = np.array(range(1,101))

x_pred = np.array([101,102])



x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_pred = torch.tensor(x_pred, dtype=torch.float32).unsqueeze(1).to(DEVICE)




print(x.dtype)  # torch.float32
print(x.size(), y.size())     # torch.Size([100, 1]) torch.Size([100, 1])



# x = torch.FloatTensor(x).to(DEVICE) 
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE) 

###### 아래가 권장 문법 ######
x = torch.tensor(x, dtype=torch.float).to(DEVICE) 
y = torch.tensor(y, dtype=torch.float).to(DEVICE) 


x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.3, # 생략가능, 디폴트: 0.25
                            shuffle=True,    # 디폴트: True
                            random_state=222,
                            )


x_mean = torch.mean(x_train)
x_std = torch.std(x_train)
x_scaled = (x_train - x_mean) / x_std
print('scaled x:', x_scaled)

# scaled x: tensor([[ 1.6181],
#         [ 1.7539],
#         [-1.5055],
#         [ 1.4822],
#         [-1.6074],
#         [ 0.5995],
#         [ 1.6520],
#         [-0.2154],
#         [-0.3172],
#         [ 0.1581],
#         [-0.6567],
#         [-0.1135],
#         [-0.3851],
#         [ 0.5316],
#         [ 0.1242],
#         [ 0.3958],
#         [-0.8604],
#         [-0.5888],
#         [ 0.8372],
#         [-0.4530],
#         [-0.1814],
#         [ 0.2939],
#         [-0.8944],
#         [-1.3697],
#         [-0.9623],
#         [-1.2339],
#         [ 0.0563],
#         [ 1.4483],
#         [-0.5209],
#         [ 0.5655],
#         [ 1.0409],
#         [-0.6228],
#         [ 0.0223],
#         [-0.7925],
#         [ 0.7693],
#         [ 1.0748],
#         [ 0.1921],
#         [ 1.2446],
#         [-1.4376],
#         [ 1.1427],
#         [-1.4716],
#         [ 0.7353],
#         [-1.1660],
#         [-1.2000],
#         [-0.8265],
#         [ 1.5501],
#         [ 0.2600],
#         [ 1.6860],
#         [ 0.4637],
#         [-1.0981],
#         [ 1.5841],
#         [-1.3018],
#         [ 0.3279],
#         [-1.0302],
#         [-0.1474],
#         [ 0.0902],
#         [-0.4191],
#         [ 1.2106],
#         [ 1.0069],
#         [-0.9283],
#         [-1.0642],
#         [ 1.3804],
#         [-0.7246],
#         [ 0.8032],
#         [-1.3358],
#         [ 1.3125],
#         [ 0.6334],
#         [-1.1321],
#         [-0.0795],
#         [-1.4037]], device='cuda:0')



model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1),
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.15)

def train(model, criterion, optimizer, x_scaled, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_scaled)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 700
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_scaled, y_train)
    print('epoch: {}, loss: {}'.format(epoch, loss))
print('----------------------------------------------------')    


def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print('Final Loss:', loss2)

x_pred = (x_pred - x_mean) / x_std

result = model(x_pred)
print('예측값:\n', result.tolist())
# print('예측값:\n', result.detach())           # 2개 이상 툴력할때. device='cuda:0' 까지 같이 나옴
# print('예측값:\n', result.detach().cpu().numpy())  # numpy 연산은 gpu가 없음. 그래서 cpu 씀

# Final Loss: 3160808.25
# 예측값:
#  [[102.0], [102.99999237060547]]
