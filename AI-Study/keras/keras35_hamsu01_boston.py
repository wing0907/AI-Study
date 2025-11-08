import sklearn as sk
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston
import time

dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


print(x.shape, y.shape)

# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


# model = Sequential()

# model.add(Dense(32, input_dim = 13, activation='relu'))
# model.add(Dropout(0.3))     # 상위 layer의 30% 가 빠지는 것
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.3))     
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))     
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.1))     
# model.add(Dense(1, activation='linear'))


input1 = Input(shape=(13,))
dense1 = Dense(32, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation='relu')(drop1)
dense3 = Dense(32, activation='relu')(dense2)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(32, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense4)
dense5 = Dense(32, activation='relu')(drop3)
drop4 = Dropout(0.1)(dense5)
output1 = Dense(1)(drop4)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()




