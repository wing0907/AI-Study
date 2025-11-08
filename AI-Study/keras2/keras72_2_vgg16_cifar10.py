import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import tensorflow as tf
import random
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from tensorflow.keras.applications import DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool2D
from tensorflow.keras.datasets import cifar10

import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


vgg16 = VGG16(
    include_top=False,
    input_shape=(32,32,3),
)


#  1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000, )

print(np.unique(y_train, return_counts=True))

print(pd.value_counts(y_test))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
#       dtype=int64))

######### 스케일링 2. 정규화 (많이 사용함) 데이터를 0에서 1로 만드는 것
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0  0.0
print(np.max(x_test), np.min(x_test))   # 1.0  0.0

print(x_train.shape, x_test.shape)      # (50000, 32, 32, 3) (10000, 32, 32, 3)

# x_train = x_train.reshape(50000, 32*32, 3)
# x_test = x_test.reshape(10000, 32*32, 3)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (50000, 10) (10000, 10)


vgg16.trainable=False       # 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())  # False하고 요로케 flatten 붙이면 됨
model.add(GlobalAvgPool2D())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

model.summary()

# 기존 모델
# model = Sequential()
# model.add(Conv1D(filters=128, kernel_size=2,
#                  padding='same', input_shape=(32*32,3), activation='relu'))
# model.add(Conv1D(128, 2))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
# model.add(Dense(units=10, activation='softmax'))
# model.summary()


# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

################ mcp 세이브 파일명 만들기 ################
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   

# path = './_save/keras39/'
# filename = '.hdf5'
# filepath = "".join([path, 'k39_0333',filename])
# #######################################################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath=filepath
# )

start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],)
end = time.time()

# 4. 평가, 예측
loss= model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_pred = model.predict(x_test)
# print(y_pred)

# y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
y_test = y_test.values
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('loss: ', round(loss[0], 4))
print('acc:' , round(acc, 4))
print('걸린시간:', end - start, '초')

### 실습 ###
# 비교할 것
# 이전의 본인이 한 최상의 결과가
# 1. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 2. 가중치를 동결하고 훈련시켰을 때, trainable=False
# 시간까지 비교할 것

### 추가 ###
# Flatten 과 GAP 비교

###########
# cifar10
# cifar100
# horse
# rps
# kaggle cat dog
# men women
###########


# ================ 기존 결과값 (epoch최소100이상) ================
# loss :  1.0032650232315063
# acc: 0.7549
# 걸린시간: 423.5881996154785 초


# ================ 가중치 동결 + Flatten , epochs=5 ================
# loss :  1.4446048736572266
# acc: 0.5029
# 걸린시간: 29.327091693878174 초


# ================ 가중치 비동결 + Faltten , epochs=5 ================
# loss:  3.9174
# acc: 0.3177
# 걸린시간: 70.45867204666138 초


# ================ 가중치 동결 + GAP , epochs=5 ================
# loss:  1.4446
# acc: 0.5029
# 걸린시간: 30.057910919189453 초


# ================ 가중치 비동결 + GAP , epochs=5 ================
# loss:  4.0834
# acc: 0.2926
# 걸린시간: 70.5808162689209 초

