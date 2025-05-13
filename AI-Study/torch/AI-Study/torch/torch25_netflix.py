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

path = 'C:\Study25\_data\Kaggle\\netflix\\'
train_csv = pd.read_csv(path+'train.csv')
# test_csv =pd.read_csv(path+'test.csv')

# print(train_csv)
# print(train_csv.info())
# print(train_csv.describe())
# print(train_csv.shape) #(967, 6) => n, 30, 3(data = train_csv.iloc[:,1:4])
# exit()
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
        self.train_csv = df
        
        self.x = self.train_csv.iloc[:,1:4].values.astype(np.float32)
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x, axis=0) - np.min(self.x, axis=0)) # MinMaxScaler
        self.y = self.train_csv['Close'].values.astype(np.float32) # 토치는 float32라서
        self.timesteps = timesteps
        #(10, 1) => (8, 3, 1) 전체 - timestep +1
        #(967, 3) => (n, 30 ,3)
    def __len__(self):
        return len(self.x) - self.timesteps # 행 - TimeSteps
    
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

#3. 컴파일, 훈련
criterion= nn.MSELoss()
from tqdm import tqdm
from torch.optim import Adam
optim = Adam(params=model.parameters(), lr = 0.001)

for epoch in range(1, 201):
    iterator = tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()
        hypothesis = model(x.to(DEVICE))
        loss = nn.MSELoss()(hypothesis, y.to(DEVICE))
        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch: {epoch}, loss: {round(loss.item(),5)}')

## save ##
save_path = 'C:\study25\_save\\torch/'
torch.save(model.state_dict(), save_path+'t25_netflix.pth')