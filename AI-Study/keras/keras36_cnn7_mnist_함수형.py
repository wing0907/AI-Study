import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Input
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


######### 스케일링 1. MinMaxScaler()
# x_train = x_train.reshape(60000, 28*28) # (60000, 784)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
# print(x_train.shape, x_test.shape)      # (60000, 784) (10000, 784)
# print(np.max(x_train), np.min(x_train)) #  255 0
# print(np.max(x_test), np.min(x_test))   # 255 0

# scaler = MinMaxScaler() # (이미지에서는 안씀)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train)) # 1.0    0.0
# print(np.max(x_test), np.min(x_test))   # 24.0   0.0 (24가 나오면 안됨)
# print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

######### 스케일링 2. 정규화 (많이 사용함) 데이터를 0에서 1로 만드는 것  activation='relu'
# x_train = x_train/255.
# x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) # 1.0  0.0
# print(np.max(x_test), np.min(x_test))   # 1.0  0.0

######### 스케일링 3. 정규화2       (많이 사용함) 데이터를 -1에서 1로 만드는 것 activation='tanh' & BatchNormalization()
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5
# print(np.max(x_train), np.min(x_train)) # 1.0 -1.0
# print(np.max(x_test), np.min(x_test))   # 1.0 -1.0
# activation='tanh' 사용하기
# tanh 함수는 쌍곡 탄젠트 함수로, 신경망에서 자주 사용하는 비선형 활성화 함수입니다.
# 출력  범위	-1 ~ 1
# 중심	0을 중심으로 대칭


#  x reshape -> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # (10000, 28, 28, 1)

print(x_train.shape, x_test.shape)      # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (60000, 10) (10000, 10)

# model = Sequential()
# model.add(Conv2D(128, (3,3), strides=2, input_shape=(28,28,1)))
# model.add(Conv2D(filters=128, kernel_size=(3,3)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(512, (3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(units=64, input_shape=(128,)))
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
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras36_cnn5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k36_', date, '_', filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model2.fit(x_train, y_train, epochs=500, batch_size=512, # batch는 행이다!!!!!!!!
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

# acc: 0.9839
# 걸린시간: 1959.4249949455261 초


##########################################################
#CNN(Convolutional Neural Network) 합성곱 신경망
#이미지나 시계열 데이터 같은 공간적 구조가 있는 데이터
#inpyt_Layer = 일반적으로 이미지 데이터
#활성화 함수Activation ='ReLU'가 일반적(비선형 구조)
#풀링층(Pooling Layer) 디멘션 정보를 줄여 차원을 축소,ex) MaxPooling: 최댓값만 선택
#과적합 방지와 학습 속도

#완전 연결층(Fully Connected Layer, Dense Layer)
# Input Image (28x28x1)
# ↓
# Conv2D (필터 적용) 이미 숫자로 되어 있는 이미지에서 의미 있는 “특징(feature)”을 추출하는 역할
# ↓
# ReLU
# ↓
# MaxPooling2D
# ↓
# Conv2D
# ↓
# ReLU
# ↓
# MaxPooling2D
# ↓
# Flatten 다차원(2D 또는 3D) 텐서를 1차원 벡터로 "펼치는" 작업
# ↓
# Dense
# ↓
# Dropout (선택)
# ↓
# Dense (출력, softmax)

# 초기 Conv 층:
# → 밝기 변화, 모서리, 선, 점 같은 기초 패턴 인식

# 중간 Conv 층:
# → 눈, 귀, 다리처럼 부분 구조 인식

# 깊은 Conv 층:
# → "고양이", "자동차"처럼 전체 의미 인식

"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터

data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()
training_images = training_images.reshape(60000,28,28,1)
"""
#2. 모델 구성

#3. 컴파일, 훈련

#4. 평가, 예측