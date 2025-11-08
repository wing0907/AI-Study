import numpy as np 
import pandas as pd
import sklearn as sk
import time as time
import ssl as ssl

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype

ssl._create_default_https_context = ssl._create_unverified_context

datasets = fetch_covtype(data_home='./fresh_data')

x = datasets.data #(581012, 54) 
y = datasets.target #(581012,)

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))

#데이터 확인 

from sklearn.preprocessing import OneHotEncoder,StandardScaler

y = y.reshape(-1, 1) 
encoder = OneHotEncoder(sparse=False) # 메트릭스형태를 받기때문에 n,1로 reshape하고 해야 한다.
y = encoder.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test =  train_test_split(
    x,y, test_size = 0.2, random_state=111, 
    stratify=y
)

#모델 생성

model = Sequential()
model.add(Dense(128, input_dim=54, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,  activation='softmax'))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)



import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:44.718308
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")     # string 문자열
print(date)     # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras28_mcp/10_fetch_covtype/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_', date, '_', filename])



mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, 
          epochs=1000,
          batch_size=2048,
          validation_split=0.2,
          verbose=1,
        #   class_weight=class_weight,
          callbacks=[es, mcp])

loss = model.evaluate(x_test, y_test)

print('loss:', loss[0])
print('acc:', loss[1])



# loss: 0.2840808928012848
# acc: 0.8889529705047607