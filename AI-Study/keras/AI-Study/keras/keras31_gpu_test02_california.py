# 17_2 copy
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
import numpy as np

# 1. 데이터
dataset = fetch_california_housing()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        test_size=0.1,
                        random_state=195) #21 , 6, 36
                                                  

# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

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
print('=====================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test]) # 원래의 y값과 예측된 y값의 비교
print('loss:', loss)
# print('[x]의 예측값:', results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어:', r2)   
rmse = np.sqrt(mean_squared_error(y_test, results))
print("rmse", rmse)
print("걸린시간 :", end - start, '초')

# cpu 사용결과
# loss: 0.46370425820350647
# r2 스코어: 0.6598907978884015
# rmse 0.6809583737030869
# 걸린시간 : 151.0214719772339 초

# gpu 사용결과
# loss: 0.432920902967453
# r2 스코어: 0.6824692243321809
# rmse 0.6579672625364222
# 걸린시간 : 152.24731397628784 초