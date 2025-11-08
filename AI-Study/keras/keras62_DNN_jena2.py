### [실습2]
### y 값을 144개 땡겨와서 진행 진행
# 56.copy
# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# 
# x_prd = 31.12.2016 00:10:00
#         ~ 01.01.2017 00:00:00(144개)의 exel, O열 wd(deg) > CSV 파일
# 평가지표 RMSE

##########################################################################
#0. 준비
##########################################################################
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization, LSTM, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import datetime as datetime
import pandas as pd
import numpy as np
import time

##########################################################################
#1. 데이터
##########################################################################
### 경로
path_D = 'c:/Study25/_data/kaggle/jena/'
path_W = 'c:/Study25/_data/kaggle/jena/weights/'
path_S = 'c:/Study25/_data/kaggle/jena/save/'
path_M = 'c:/Study25/_data/kaggle/jena/MCP/'
path_Sub = 'c:/Study25/_data/kaggle/jena/sub/'
#####################################

T1 = time.time()

x_csv = pd.read_csv(path_D + 'x_trn_0620_8.csv', index_col=False)
y_csv = pd.read_csv(path_D + 'y_trn_0620_8.csv', index_col=False)

""" print(x.shape) (420048, 13) """
""" print(x.info()) non-null """
""" print(y.shape) (420048, 1) """
""" print(y.info()) non-null """

y_csv['wd (deg)'] = y_csv['wd (deg)'].shift(-1)

x = np.array(x_csv)
y = np.array(y_csv)

x = x.reshape(-1,144*5)
y = y.reshape(-1,144)

# print(x.shape) (2919, 720)
# print(y.shape) (2919, 144)


x_trn = x[:-1,:]
x_prd = x[-1:,:]
y_trn = y[:-1,]
y_tru = y[-2:-1,]

# print(x.shape) (2917, 144, 13)
# print(y.shape) (2917, 144)

T2 = time.time()
print('전처리 1 :', T2 - T1)

RS = 42

x_trn, x_tst, y_trn, y_tst = train_test_split(x_trn,y_trn,
                                              train_size=0.75,
                                              shuffle=True,
                                              random_state=RS
                                              )

x_trn, x_val, y_trn, y_val = train_test_split(x_trn,y_trn,
                                              train_size=0.1,
                                              shuffle=True,
                                              random_state=RS
                                              )

##########################################################################
#2. 모델 구성
##########################################################################
### 불러오기
# model = load_model(path_S + 'jena_0619_0.h5')
#####################################

model = Sequential()

model.add(Dense(50, input_dim = x_trn.shape[1],
               activation='relu'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Dense(100, activation='sigmoid'))
# model.add(Dropout(0.2))

# model.add(BatchNormalization())
# model.add(Dense(150, activation='sigmoid'))

model.add(Dense(144, activation='sigmoid'))

E, B, P = (100000, 20, 10)

''' rmse : 68.0954063526045 '''

#####################################
### saveNum
date = datetime.datetime.now()
date = date.strftime('%m%d')
saveNum = f'{date}_0'

""" radian loss
import tensorflow as tf

def angle_to_vector(angle_deg):
    Rad = tf.constant(np.pi / 180., dtype=tf.float32)
    angle_rad = angle_deg*Rad
    return tf.stack([tf.cos(angle_rad), tf.sin(angle_rad)], axis=1)

def angular_loss(y_tru, y_prd):
    vec_tru = angle_to_vector(y_tru)
    vec_prd = angle_to_vector(y_prd)
    
    dot = tf.reduce_sum(vec_tru*vec_prd, axis=-1)
    loss = 1.0 - dot
    return tf.reduce_mean(loss)

"""
#####################################
# Callbacks 설정
ES = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=P,
                   restore_best_weights=False)

MCP = ModelCheckpoint(monitor='val_loss',
                      mode='min',
                      filepath= "".join([path_M,'/','MCP_',saveNum,'_','.h5']),
                      save_best_only=True)

##########################################################################
#3. 컴파일 훈련
##########################################################################
model.compile(loss ='mse',
              optimizer='adam'
              )

### 가중치 불러오기
# model.load_weights(path_W + '0619_0.h5')

model.fit(x_trn, y_trn,
          epochs = E,
          batch_size= B,
          verbose = 1,
          validation_data = (x_val, y_val),
          callbacks = [ES, MCP])

model.save(path_S + f'jena_{saveNum}.h5')
model.save_weights(path_W + f'{saveNum}.h5')

T5 = time.time()
##########################################################################
#4. 평가 예측
##########################################################################
# loss = model.evaluate(x_tst, y_tst)

x_prd = pd.read_csv(path_D + 'x_prd_0620_8.csv', index_col=False)
y_tru = pd.read_csv('C:/Study25/_data/kaggle/jena/' + 'y_prd_0620_8.csv', index_col=False)

x_prd = np.array(x_prd)
y_tru = np.array(y_tru)

x_prd = x_prd.reshape(-1,144*5)
y_tru = y_tru.reshape(-1,144)

y_prd = model.predict(x_prd)

y_tru = y_tru*360.
y_prd = y_prd*360.

rmse = np.sqrt(mean_squared_error(y_tru, y_prd))

# print('save :', saveNum)
print('rmse :', rmse)

#####################################
### 파일 송출 조건
# column = DateTime / wd(deg)
# xy_csv = pd.read_csv(path_D + 'jena_climate_2009_2016.csv', index_col=0)
# index = xy_csv.index[-144:]
# # print(index.shape)

# # 파일명 : jena_박재익_submit.csv
# sub = pd.DataFrame({
#     'Date Time' : index,
#     'wd (deg)': y_prd.flatten()
# })

# sub.to_csv(path_Sub + f'jena_박재익_submit_{saveNum}.csv', index=False)