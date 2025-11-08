# 17_2 copy
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
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
                                                  
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


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

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',                    # 최대값 max, 알아서 찾아줘: auto
    patience=20,                     # 이만큼 참을 거다. local minimal, global minimal
    restore_best_weights=True,       #EarlyStopping 의 Default는 False. 최적의 weight 때 멈춘다. 최소지점 save.
)





##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
hist = model.fit(x_train,y_train, epochs = 400, batch_size =32,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es],
          
          )





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

# scaler 안쓴거
# loss: 0.4433603584766388
# r2 스코어: 0.6748123114429134
# rmse 0.6658530923243134


# scaler = MinMaxScaler()
# loss: 0.27609217166900635
# r2 스코어: 0.79749705009043
# rmse 0.525444742007467


# scaler = StandardScaler()
# loss: 0.29656121134757996
# r2 스코어: 0.7824838062793968
# rmse 0.5445743364658828


# scaler = MaxAbsScaler()
# loss: 0.3517691493034363
# r2 스코어: 0.741990954820531
# rmse 0.5931012498964875


# scaler = RobustScaler()
# loss: 0.2826680541038513
# r2 스코어: 0.7926738908439244
# rmse 0.531665368444084
