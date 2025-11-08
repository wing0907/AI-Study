import numpy as np
from tensorflow.keras.datasets import cifar10
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

# 1. Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# Scaling
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])
print(x_train.shape, y_train.shape)    # (50000, 3072) (50000, 1)
print(x_test.shape, y_test.shape)      # (10000, 3072) (10000, 1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)      # (50000, 10) (10000, 10)



model = Sequential()
model.add(Dense(512, input_dim=3072, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax'))
model.summary()


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras40_dnn/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k40_3', date, '_', filename])

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
print('loss : ', loss[0])
print('acc : ', loss[1])

y_pred = model.predict(x_test)
print(y_pred)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:' , round(acc, 4))
print('걸린시간:', end - start, '초')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x_test.reshape(-1, 32, 32, 3)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# acc: 0.5342
# 걸린시간: 104.16143369674683 초
