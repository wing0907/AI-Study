# copy from 36_3
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#  실무상에서는 폴더에서 이미지를 직접 수치화 해야 한다

# print(x_train)
# print(x_train[0])
# print(y_train[0]) # 5
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
    #   dtype=int64))
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892

######### 스케일링 2. 정규화 (많이 사용함) 데이터를 0에서 1로 만드는 것
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0  0.0
print(np.max(x_test), np.min(x_test))   # 1.0  0.0


#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # (10000, 28, 28, 1)

print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

# model = Sequential()
# model.add(Conv2D(128, (2,2), strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D())                                               # MaxPooling 한방 때리면 반띵됨
# model.add(Conv2D(filters=512, kernel_size=(2,2)))
# model.add(MaxPooling2D())                                               # MaxPooling 한방 때리면 반띵됨
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(units=64, input_shape=(128,)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(units=10, activation='softmax'))
# model.summary()

input1 = Input(shape=(28,28,1))
conv2d1 = Conv2D(128, (3,3), strides=2)(input1)
conv2d2 = Conv2D(filters=128, kernel_size=(3,3))(conv2d1)
batch1 = BatchNormalization()(conv2d2)
drop1 = Dropout(0.2)(batch1)
conv2d3 = Conv2D(512, (3,3), activation='tanh')(drop1)
flat1 = Flatten()(conv2d3)
dense1 = Dense(units=128, activation='tanh')(flat1)
batch2 = BatchNormalization()(dense1)
drop2 = Dropout(0.2)(batch2)
dense2 = Dense(64, input_shape=(128,))(drop2)
output1 = Dense(units=10, activation='softmax')(dense2)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()


# 3. 컴파일, 훈련
model2.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

################ mcp 세이브 파일명 만들기 ################
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   

path = './_save/keras32/'
filename = '.hdf5'
filepath = "".join([path, 'k42_2',filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model2.fit(x_train, y_train, epochs=20, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

# 4. 평가, 예측
loss= model2.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_pred = model2.predict(x_test)
print(y_pred)

# y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
y_test = y_test.values
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:' , round(acc, 4))
print('걸린시간:', end - start, '초')

# acc: 0.9028
# 걸린시간: 267.4948709011078 초

aaa = 3
print(y_train[aaa])

import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
plt.imshow(x_train[aaa])
plt.show()


# 예측값과 정답값 준비 (예시용)
y_pred = model2.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # 분류 모델일 경우

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x_test.reshape(-1, 28, 28, 1)  # x_test가 (N, 64)인 경우만 OK

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred_labels[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# acc: 0.8569
# 걸린시간: 165.50574111938477 초