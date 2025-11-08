import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch:', )

# 1. 데이터
path = 'C:\Study25\_data\dacon\따릉이\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isna().sum())     # 위 함수와 똑같음
train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
y = train_csv['count']  



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=222,
    # stratify=y,
)

print(x_train.shape, x_test.shape)  # (1062, 9) (266, 9)
print(y_train.shape, y_test.shape)  # (1062,) (266,)

# print(np.unique(y_train))
# exit()

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)



print(x_train.dtype)                # torch.float32
print(x_train.shape, y_train.shape) # torch.Size([397, 10]) torch.Size([397, 1])
print(type(x_train))                # <class 'torch.Tensor'>

# exit()

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(9, 64),
    nn.ReLU(),
    nn.Linear(64, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
        # nn.SiLU(),
    nn.ReLU(),
    nn.Linear(32, 1),
    # nn.Softmax(),
).to(DEVICE)

# 3. 컴파일, 훈련
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss() # Sparse Categorical Entropy. 원핫과 소프트맥스가 포함.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.08)

def train(model, criterion, optimizer, x, y):
    # model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss =criterion(hypothesis, y)

    loss.backward()
    optimizer.step()

    return loss.item()


epochs = 800
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs: {}, loss: {}'.format(epoch, loss))  # verbose
print('----------------------------------------------------')


# 4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
    return loss2.item()
final_loss = evaluate(model, criterion, x_test, y_test)
print('Final Loss:', final_loss)


######################################################
y_predict = model(x_test)
# print(type(y_predict))

# y_predict = y_predict.detach()
# print(y_predict)
# y_predict = y_predict.cpu()
# print(y_predict)
# print(type(y_predict))
# y_predict = y_predict.numpy()
# print(type(y_predict))
y_predict = np.round(y_predict.detach().cpu().numpy())



y_test = y_test.detach().cpu().numpy()

r2 = r2_score(y_test, y_predict)
print('r2:', r2)


# Final Loss: 2375.72216796875
# r2: 0.6775024128723259







exit()
#######################################################
y_predict = model(x_test).cpu() # acc 빼기위함
y_predict_cls = (y_predict >= 0.5).int().numpy()
y_true = y_test.cpu().numpy().astype(int)

acc = accuracy_score(y_true, y_predict_cls)
print('acc:', acc)
# Final Loss: 0.8519402146339417
# acc: 0.9590643274853801

