# https://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd
print(np.__version__)       # 1.23.0
print(pd.__version__)       # 2.2.3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle/bike/'         # 상대경로
# path = '.\_data\kaggle\\bike\\'       #\b \n \a \' 는 예약어. 예약어 사용시 노란줄 표출
# path = '.\\_data\\kaggle\\bike\\'     # \n 줄바꿈 
# path = './/_data//kaggle//bike//'

# path = 'c:/Study25/_data/kaggle/bike/' # 절대경로


train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv)            # [10886 rows x 11 columns]
print(train_csv.shape)      # (10886, 11)
print(test_csv.shape)       # (6493, 8)
print(sampleSubmission_csv.shape)       #(6493, 1)

# exit()

print(train_csv.columns)
"""
Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')
"""

print(train_csv.info())  # 결측치 없음
print(test_csv.info())   # no Nan


x = train_csv.drop(['count', 'registered', 'casual'], axis=1)   # [10886 rows x 8 columns]
print(x)            # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)      # (10886,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=805
)

#2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

def rmse(y_test, y_predict):
       return np.sqrt(mean_squared_error(y_test, y_predict))

rmses = rmse(y_test, results)
r2 = r2_score(y_test, results)

from sklearn.metrics import mean_squared_log_error

def rmsle (y_test, y_predict):
    y_test = np.maximum(y_test, -1)     # 예측값에 -1 이하의 값이 있다면, 이를 0으로 변경
    y_predict = np.maximum(y_predict, -1)
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

rmsles = rmsle(y_test, results)

y_submit = model.predict(test_csv)
print(y_submit.shape)  # (6493, 1)

# exit()

print(sampleSubmission_csv)
sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)

sampleSubmission_csv.to_csv(path + 'sampleSubmission_0522_1648.csv')


print('[loss]:', loss)
print('[rmse]:', rmses)
print('[r2]:', r2)
print('[rmsle]', rmsles)

# 0521_1735  rs 4631 epochs 400  bs 32
# [loss]: 25915.41015625
# [rmse]: 160.98264024516772
# [r2]: 0.25544082236577903
# [rmsle] 1.2639785004368438

# 0522_1540      rs 4631 epochs 200  bs 32
# [loss]: 22327.140625
# [rmse]: 149.42269018452328
# [r2]: 0.3585331292586912
# [rmsle] 1.2636874790818073

# 0522_1549     rs 4631 epochs 300  bs 30
# [loss]: 23964.794921875
# [rmse]: 154.80566576429553
# [r2]: 0.3114827390498688
# [rmsle] 1.255391894173333

# 0522_1622     rs 805 epochs 200  bs 32
# [loss]: 20882.638671875
# [rmse]: 144.5082713139944
# [r2]: 0.2902050905729605
# [rmsle] 1.2804497064994005

# 0522_1648  rs  epochs 200  bs 32