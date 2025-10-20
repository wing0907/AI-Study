import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import warnings
warnings.filterwarnings('ignore')

random.seed(333)
np.random.seed(333)
torch.manual_seed(333)
torch.cuda.manual_seed(333)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

path = 'C:\study25\_data\kaggle\jena\\'
train_csv = pd.read_csv(path+'jena_climate_2009_2016.csv') #"T (degC)"

y_test_1 = train_csv[ (train_csv['Date Time'].str.contains("31.12.2016", regex=False)) &
    (train_csv['Date Time'] != "31.12.2016 00:00:00")]['wd (deg)'].copy()
y_test_2 = train_csv[train_csv['Date Time'].str.contains("01.01.2017")]['wd (deg)'].copy()
y_test = pd.concat([y_test_1, y_test_2], ignore_index=True) #ignore_index 인덱스를 새로 매겨줘

# 날짜 처리
train_csv['Date Time'] = pd.to_datetime(train_csv['Date Time'], format="%d.%m.%Y %H:%M:%S")
train_csv['hour'] = train_csv['Date Time'].dt.hour
train_csv['month'] = train_csv['Date Time'].dt.month
train_csv['day'] = train_csv['Date Time'].dt.day
train_csv['minute'] = train_csv['Date Time'].dt.minute
train_csv['weekday'] = train_csv['Date Time'].dt.weekday
train_csv = train_csv.drop(['Date Time'],axis=1)

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self,df,timesteps=30):
        self.train_csv = df
        self.x = self.train_csv.values.astype(np.float32)
        self.x = (self.x - np.min(self.x, axis=0)) / (np.max(self.x,axis=0)-np.min(self.x,axis=0))
        self.y = self.train_csv['wd (deg)'].values.astype(np.float32) # 토치는 float32라서
        self.timesteps = timesteps
    def __len__(self):
        return len(self.x) - self.timesteps # 행 - TimeSteps
    def __getitem__(self, index):
        x = self.x[index: index+self.timesteps]
        y = self.y[index+self.timesteps]
        return x, y
custom_Dataset = Custom_Dataset(df=train_csv, timesteps=30)
train_loader = DataLoader(custom_Dataset, batch_size=1)


for batch_idx, (xb, yb) in enumerate(train_loader):
    print('===== 배치 :', batch_idx,'=====')
    print('x :', xb)
    print('y :',yb)
    break

#2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size=19,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(32,2)
        self.relu = nn.ReLU()  
        self.tanh = nn.Tanh()
    def forward(self,x):
        x,_ = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x
    
model = RNN().to(DEVICE)     

#3. 컴파일, 훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1e-2)
def train(model, criterion, optimizer, loader):
    epoch_loss = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        model.train()
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        
        loss = criterion(hypothesis, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss/ len(loader)
def evaluate(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            epoch_loss += loss.item()
        return epoch_loss / len(loader)
EPOCH = 1
for epoch in range(1,EPOCH+1):
    loss = train(model, criterion, optimizer, train_loader)
# ## save ##
save_path = 'C:\study25\_save\\torch/'
torch.save(model.state_dict(), save_path+'t25_jenna.pth')

#4. 평가 예측


from sklearn.metrics import r2_score

y_predict = []
total_loss = 0
y_true = []

with torch.no_grad():
    model.load_state_dict(torch.load(save_path+'t25_jenna.pth',map_location=DEVICE))
    
    for x_test, y_test in train_loader:
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE))
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())
        
        loss = criterion(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)
        
print(total_loss)

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
            
            
            
        
        
        
        