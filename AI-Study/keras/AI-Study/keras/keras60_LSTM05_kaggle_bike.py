import numpy as np
import pandas as pd
# import sklearn as sk
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, BatchNormalization, Dropout, Input, LSTM, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=361,
    )

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))   
print(np.min(x_test), np.max(x_test))     

print(x_train.shape, y_train.shape) # (9797, 10) (9797,)
print(x_test.shape, y_test.shape)   # (1089, 10) (1089,)

x_train = x_train.reshape(-1,10,1)
x_test = x_test.reshape(-1,10,1)


print(x_train.shape, y_train.shape) # (9797, 10, 1) (9797,)
print(x_test.shape, y_test.shape)   # (1089, 10, 1) (1089,)


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(10,1)))                                    
model.add(LSTM(units=512, activation='relu'))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
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
import time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_5', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=512,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.9998
# 걸린시간: 48.12139654159546 초

# r2: 0.9998
# 걸린시간: 142.92842984199524 초