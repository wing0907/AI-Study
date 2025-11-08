# dacon, 데이터 파일 별도
# http://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
path = './_data/dacon/diabetes/'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(submission_csv)             # [116 rows x 1 columns]


print(train_csv.shape)              # (652, 9)
print(test_csv.shape)               # (116, 8)
print(submission_csv.shape)         # (116, 1)

print(train_csv.info())
print(train_csv.describe())

print(test_csv.info())

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# x = x.replace(0, np.nan)

# train_csv = train_csv.replace(0, np.nan)
# train_csv = train_csv.fillna(train_csv.mean())
x = x.replace(0, np.nan)
x = x.fillna(x.mean())

print(train_csv.isna().sum()) 
print(train_csv.info()) 

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

print(test_csv.info())           

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)


print(x)                # [652 rows x 8 columns]
print(y.shape)          # (652,)


print(pd.DataFrame(y).value_counts())
# 1    228
# 0    424
print(pd.Series(y).value_counts())
# 1    228
# 0    424


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=55,                   # 33
    shuffle=True, 
)

print(x_train.shape, x_test.shape)      # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)      # (456,) (196,)


# 2. 모델구성.
from tensorflow.keras.layers import Dropout, BatchNormalization

model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(BatchNormalization())                             # Dropout 또는 BatchNormalization을 추가하면 학습 안정성 향상에 좋습니다.
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진분류는 무조건 마지막 activation='sigmoid' 이다. node는 1개


# 3. 컴파일, 훈련

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)  # 0.001 또는 0.0005
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,       
          )
end = time.time()


import tensorflow as tf
gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)   


y_predict = model.predict(x_test)         
print(y_predict[:10])
y_predict = np.nan_to_num(y_predict)
y_predict = np.round(y_predict).astype(int)
print(y_predict[:10])

from sklearn.metrics import accuracy_score              
accuracy_score = accuracy_score(y_test, y_predict)
print("acc_score : ", accuracy_score)


y_submit = model.predict(test_csv) 
y_submit = np.round(y_submit)
submission_csv['Outcome'] = y_submit

print("loss : ", results[0])            
print("acc : ", round(results[1], 4))            

print("걸린시간 :", end - start, 'seconds')

# cpu 사용결과
# acc_score :  0.7244897959183674
# loss :  0.7432868480682373
# acc :  0.7245
# 걸린시간 : 3.7523603439331055 seconds

# gpu 사용결과
# loss :  0.7353930473327637
# acc :  0.7143
# 걸린시간 : 10.611941814422607 seconds