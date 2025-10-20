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

path = 'c:/study25/_data/kaggle/netflix/'
train_csv = pd.read_csv(path + 'train.csv')
print(train_csv)  # [967 rows x 6 columns]
print(train_csv.info())
print(train_csv.describe())

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 967 entries, 0 to 966
Data columns (total 6 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Date    967 non-null    object
 1   Open    967 non-null    int64
 2   High    967 non-null    int64
 3   Low     967 non-null    int64
 4   Volume  967 non-null    int64
 5   Close   967 non-null    int64
dtypes: int64(5), object(1)
memory usage: 45.5+ KB
None
             Open        High         Low        Volume       Close
count  967.000000  967.000000  967.000000  9.670000e+02  967.000000
mean   223.923475  227.154085  220.323681  9.886233e+06  223.827301
std    104.455030  106.028484  102.549658  6.467710e+06  104.319356
min     81.000000   85.000000   80.000000  1.616300e+06   83.000000
25%    124.000000  126.000000  123.000000  5.638150e+06  124.000000
50%    194.000000  196.000000  192.000000  8.063300e+06  194.000000
75%    329.000000  332.000000  323.000000  1.198440e+07  327.500000
max    421.000000  423.000000  413.000000  5.841040e+07  419.000000
'''

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

data = train_csv.iloc[:,1:4]
data['종가'] = train_csv['Close']
print(data)

hist = data.hist()
plt.show()

from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, df, timesteps=30):
        self.train_csv = df
        
        self.x = self.train_csv.iloc[:, 1:4].values  # numpy 형태로 변환
        self.x = (self.x - np.min(self.x, axis=0)) / \
            (np.max(self.x, axis=0) - np.min(self.x, axis=0)) # MinMaxScaler 임
        
        self.y = self.train_csv['Close'].values
        self.timesteps = timesteps
       
    # (10, 1) -> (8, 3, 1) 전체 - timestep + 1   
    # (967, 3) -> (n, 30, 3)   
    def __len__(self):
        return len(self.x) - self.timesteps   # 행 - timesteps + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.timesteps]        # 1,30,3   # x[idx : idx + timesteps]
        y = self.y[idx + self.timesteps]                         # y[idx + timesteps]
        return x, y        

custom_dataset = Custom_Dataset(df=train_csv, timesteps=30)

train_loader = DataLoader(custom_dataset, batch_size=32)

for batch_idx, (xb, yb) in enumerate(train_loader):
    print("====== 배치:", batch_idx, "="*6)
    print("x:", xb.shape)   # x: torch.Size([32, 30, 3])
    print("y:", yb.shape)   # y: torch.Size([32])
    break

# 2. 모델
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size=3,
                          hidden_size=64,
                          num_layers=3,
                          batch_first=True,
                          ) # (n, 30, 64)
        # self.fc1 = nn.Linear(in_features=30*64, out_features=32)
        self.fc1 = nn.Linear(in_features=64, out_features=32)

        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x, _ = self.rnn(x)

        # x = x.reshape(-1, 30*64)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RNN().to(DEVICE)

# 3. 컴파일, 훈련
"""
from torch.optim import Adam
optim = Adam(params=model.parameters(), lr = 0.001)

# import tqdm
from tqdm import tqdm

for epoch in range(1, 201):
    iterator = tqdm.tqdm(train_loader)
    for x, y in iterator:
        optim.zero_grad()

        # h0 = torch.zeros(5, x.shape[0], 64).to(DEVICE)    # num_layers, batch_size, hidden_
                
        hypothesis = model(x.type(torch.FloatTensor).to(DEVICE))

        loss = nn.MSELoss()(hypothesis, y.type(torch.FloatTensor).to(DEVICE))

        loss.backward()
        optim.step()
        
        iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')
"""
### save ###
save_path = './_save/torch/'
torch.save(model.state_dict(), save_path + 't25_netflix.pth')

# 4. 평가, 예측
y_predict = []
total_loss = 0
y_true = []

with torch.no_grad():
    model.load_state_dict(torch.load(save_path + 't25_netflix.pth', map_location=DEVICE))
    
    for x_test, y_test in train_loader:
        y_pred = model(x_test.type(torch.FloatTensor).to(DEVICE))
        y_predict.append(y_pred.cpu().numpy())
        y_true.append(y_test.cpu().numpy())

        loss = nn.MSELoss()(y_pred, y_test.type(torch.FloatTensor).to(DEVICE))
        total_loss += loss / len(train_loader)

print('-'*50)
print(total_loss)

from sklearn.metrics import r2_score

print(type(y_predict))
print(type(y_test))

y_predict = np.concatenate(y_predict).flatten()
y_true = np.concatenate(y_true)

r2 = r2_score(y_true, y_predict)
print('R2:', r2)
print('total loss:', total_loss.item())

