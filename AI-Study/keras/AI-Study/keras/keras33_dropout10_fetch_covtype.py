import numpy as np 
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

model.compile(loss = 'categorical_crossentropy',
              optimizer= 'adam', metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train,y_train, epochs = 10, batch_size =32,
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
loss = model.evaluate(x_test, y_test)

print('loss:', loss[0])
print('acc:', loss[1])
print("걸린시간 :", end - start, '초')

# cpu 사용결과
# loss: 0.3786824941635132
# acc: 0.8439455032348633
# 걸린시간 : 115.44336891174316 초


# gpu 사용결과
# loss: 0.37504488229751587
# acc: 0.8436270952224731
# 걸린시간 : 423.91082978248596 초