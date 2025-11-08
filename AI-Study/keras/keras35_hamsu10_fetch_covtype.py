import numpy as np 
import sklearn as sk
import time as time
import ssl as ssl

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense ,Dropout, Input
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

# model = Sequential()
# model.add(Dense(128, input_dim=54, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(7,  activation='softmax'))

input1 = Input(shape=(54,))
dense1 = Dense(128, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(128, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
output1 = Dense(7, activation='softmax')(drop2)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()