import numpy as np 
import pandas as pd
import sklearn as sk
import time as time
import ssl as ssl

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense ,Dropout, BatchNormalization, LSTM, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical

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

print(np.min(x_train), np.max(x_train))   # 0.0 711.0
print(np.min(x_test), np.max(x_test))     # 0.0 711.0

print(x_train.shape, y_train.shape) # (464809, 54) (464809, 7)
print(x_test.shape, y_test.shape)   # (116203, 54) (116203, 7)


x_train = x_train.reshape(-1,54,1)
x_test = x_test.reshape(-1,54,1)


print(x_train.shape, y_train.shape) # (464809, 54, 1) (464809, 7)
print(x_test.shape, y_test.shape)   # (116203, 54, 1) (116203, 7)



model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(54,1)))                                    
model.add(Dense(512, activation='relu'))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=7, activation='softmax'))
model.summary()


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_10', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=20, batch_size=512,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.625
# 걸린시간: 413.3227307796478 초


# r2: 0.1799
# 걸린시간: 2006.6968092918396 초