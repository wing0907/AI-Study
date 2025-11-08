import sklearn as sk
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=55,                   # 33
    shuffle=True, 
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)      # (309, 10) (133, 10)
print(y_train.shape, y_test.shape)      # (309,) (133,)

x_train = x_train.reshape(-1,5,2,1)
x_test = x_test.reshape(-1,5,2,1)


print(x_train.shape, y_train.shape) # (309, 5, 2, 1) (309,)
print(x_test.shape, y_test.shape)   # (133, 5, 2, 1) (133,)



model = Sequential()
model.add(Conv2D(128, (1,1), strides=1, padding='same', input_shape=(5,2,1)))                                    
model.add(Conv2D(filters=512, kernel_size=(1,1)))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))
model.summary()


model.compile(loss= 'mse', optimizer='adam')
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
filepath = "".join([path, 'k41_1', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

  
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: -2.2102
# 걸린시간: 12.033121347427368 초