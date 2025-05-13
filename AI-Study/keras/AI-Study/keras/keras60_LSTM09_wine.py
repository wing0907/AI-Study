from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import time

import sklearn as sk

from keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from tensorflow.keras.utils import to_categorical

dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



print(np.min(x_train), np.max(x_train))   # 0.0 711.0
print(np.min(x_test), np.max(x_test))     # 0.0 711.0

print(x_train.shape, y_train.shape) # (142, 13) (142,)
print(x_test.shape, y_test.shape)   # (36, 13) (36,)



x_train = x_train.reshape(-1,13,1)
x_test = x_test.reshape(-1,13,1)


print(x_train.shape, y_train.shape) # (142, 13, 1) (142,)
print(x_test.shape, y_test.shape)   # (36, 13, 1) (36,)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (142, 3) (36, 3)


model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(13,1)))                                    
model.add(LSTM(units=512, activation='relu'))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=3, activation='softmax'))
model.summary()


model.compile(loss= 'categorical_crossentropy', optimizer='adam',metrics=['acc'])
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
filepath = "".join([path, 'k41_9', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=24,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.9995
# 걸린시간: 24.200477123260498 초

# r2: 0.9086
# 걸린시간: 48.801284313201904 초