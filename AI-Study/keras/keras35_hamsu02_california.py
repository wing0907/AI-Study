# 17_2 copy
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3
import pandas as pd

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
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
# model = Sequential()
# model.add(Dense(100, input_dim=8, activation='relu'))
# model.add(Dropout(0.2))     
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.2))     
# model.add(Dense(800, activation='relu'))
# model.add(Dropout(0.2))     
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.2))     
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))     
# model.add(Dense(1))

input1 = Input(shape=(8,))
dense1 = Dense(100, activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(500, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(800, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(300, activation='relu')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(50, activation='relu')(drop4)
drop5 = Dropout(0.1)(dense5)
output1 = Dense(1)(drop5)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()