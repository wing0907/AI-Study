import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, BatchNormalization, Dropout, Input, Conv1D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

print(x.shape, y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



print(np.min(x_train), np.max(x_train))   # 0.0 711.0
print(np.min(x_test), np.max(x_test))     # 0.0 711.0

print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape)   # (114, 30) (114,)


x_train = x_train.reshape(-1,30, 1)
x_test = x_test.reshape(-1,30, 1)


print(x_train.shape, y_train.shape) # (455, 30, 1) (455,)
print(x_test.shape, y_test.shape)   # (114, 30, 1) (114,)


model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2,
                 padding='same', input_shape=(30,1)))                                    
model.add(Conv1D(512, 2))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()


model.compile(loss= 'binary_crossentropy', optimizer='adam')
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
filepath = "".join([path, 'k41_6', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=512,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.9462
# 걸린시간: 34.0512318611145 초

# r2: 0.9346
# 걸린시간: 281.6032373905182 초

# r2: -0.6103
# 걸린시간: 4.043838739395142 초