import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

# 2. 모델구성
"""
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(10,10,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Conv2D(16, (3,3)))
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()
"""
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 64)          640        3 x 3 x 1 + 1 x 64 = 640
#  conv2d_1 (Conv2D)           (None, 6, 6, 32)          18464      3 x 3 x 64 + 1 x 32 = 18464
#  conv2d_2 (Conv2D)           (None, 4, 4, 16)          4624       3 x 3 x 32 + 1 x 16 = 4624
#  flatten (Flatten)           (None, 256)               0
#  dense (Dense)               (None, 16)                4112       256 x 16 + 16 = 4112 입력노드 수 x 출력노드 수 + 출력노드 수(bias) = Dense파라미터 수
#  dense_1 (Dense)             (None, 16)                272        16 x 16 + 16 = 272
#  dense_2 (Dense)             (None, 10)                170        입력노드 수 + bias 1 x 출력노드 수 = Dense파라미터 수
# =================================================================
# Total params: 28,282
# Trainable params: 28,282
# Non-trainable params: 0

model = Sequential()
model.add(Conv2D(5, (2,2), strides=1, input_shape=(5,5,1)))
model.add(Conv2D(filters=4, kernel_size=(2,2)))
# model.add(Conv2D(3, (3,3)))
model.add(Flatten())
model.add(Dense(units=10))
# model.add(Dense(units=16))
# model.add(Dense(units=10, activation='softmax'))
# kernel_size는 중첩해서 하는게 맞지만 데이터가 크면 굳이 중첩해서 계산할 필요없다
# 따라서 strides를 사용한다. : 훈련시키는 보폭 default는 1
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 5)           25

#  conv2d_1 (Conv2D)           (None, 3, 3, 4)           84

#  flatten (Flatten)           (None, 36)                0

#  dense (Dense)               (None, 10)                370

# =================================================================
# Total params: 479
# Trainable params: 479
# Non-trainable params: 0

model = Sequential()
model.add(Conv2D(64, (3,3), strides=2, input_shape=(28,28,1)))
model.add(Conv2D(filters=4, kernel_size=(3,3)))
model.add(Conv2D(3, (3,3)))
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()