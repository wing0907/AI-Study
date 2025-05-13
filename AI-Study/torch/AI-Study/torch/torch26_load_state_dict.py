import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

path = 'C:\study25\_data\kaggle\\netflix\\'
train_csv = pd.read_csv(path+'train.csv')
# test_csv =pd.read_csv(path+'test.csv')

# print(train_csv)
# print(train_csv.info())
# print(train_csv.describe())

# import matplotlib.pyplot as plt
# data = train_csv.iloc[:,1:4]
# data['종가'] = train_csv['Close']
# print(data)

# hist = data.hist()
# plt.show()
# exit()

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self,df ,timesteps=30):
        self.train_csv = train_csv
        self.x = self.train_csv.iloc[:,1:4].values.astype(np.float32)
        self.x = (self.x -np.min(self.x, axis=0))/(np.max(self.x, axis=0)-np.min(self.x, axis=0)) # MinMaxScaler
        self.y = self.train_csv['Close'].values.astype(np.float32)
        self.timesteps = timesteps
        #(10, 1) => (8, 3, 1) 전체 - timestep +1
        #(967, 3) => (n, 30 ,3)
    def __len__(self):
        return len(self.x) - 30 # 행 - TimeSteps
    
    def __getitem__(self, index):
        x = self.x[index: index+self.timesteps] # x[idx : idx+타입스텝스]
        y = self.y[index+self.timesteps] # y[idx+타임스텝스]
        return x, y

custom_Dataset = Custom_Dataset(df =train_csv, timesteps=30)
train_loader = DataLoader(custom_Dataset, batch_size=32)

for batch_idx, (xb, yb) in enumerate(train_loader):
    print('===== 배치 :', batch_idx,'=====')
    print('x :', xb)
    print('y :',yb)
    break

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x,_ = self.rnn(x)
        # x = x.reshape(-1,x.shape[1]*64)
        x= x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = RNN().to(DEVICE)

criterion= nn.MSELoss()
# import tqdm
# from torch.optim import Adam
# optimizer = Adam(params=model.parameters(), lr = 0.001)

# for epoch in range(1, 201):
#     iterator = tqdm.tqdm(train_loader)
#     for x, y in iterator:
#         optimizer.zero_grad()
        
#         hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))
#         loss= criterion(hypothesis, y.unsqueeze(1))
        
#         loss.backward()
#         optimizer.step() # w = x -lr*로스를 웨이트로 미분한값 
#         iterator.set_description(f'epoch :{epoch} loss: {loss}')

# ## save ##
save_path = './_save/torch/'
# torch.save(model.state_dict(), save_path+'t25_netflix.pth')

#4. 평가 예측

y_predict = []
total_loss = 0
y_true = []

with torch.no_grad():
    model.load_state_dict(torch.load(save_path+'t25_netflix.pth',map_location=DEVICE))
    
    for x_test, y_test in train_loader:
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE))
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())
        
        loss = criterion(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)
        
print(total_loss)
    
from sklearn.metrics import r2_score

print(type(y_predict))
print(type(y_true))

# print((y_true))
# exit()
y_predict = np.concatenate(y_predict)
y_true =  np.concatenate(y_true)
# 안전하게 배열로 전환
# y_predict = np.array(y_predict).flatten()

r2 = r2_score(y_true, y_predict)
print('R2 :', r2)
print('total loss :', total_loss.item())