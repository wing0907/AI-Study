# copy from 58_1 실습 LSTM 넣기
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Reshape, LSTM
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
# # print(np.max(x_train), np.min(x_train)) # 1.0    0.0
# # print(np.max(x_test), np.min(x_test))   # 24.0   0.0 (24가 나오면 안됨)
# # print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

######## 스케일링 2. 정규화 (많이 사용함) 데이터를 0에서 1로 만드는 것
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0  0.0
print(np.max(x_test), np.min(x_test))   # 1.0  0.0

######### 스케일링 3. 정규화2       (많이 사용함) 데이터를 -1에서 1로 만드는 것
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
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

model = Sequential()
# model.add(Dense(100, input_shape=(28, 28))) # 이럴 경우 (N, 28, 100)이기 때문에 다음 Conv2D (N, 28, 28, 1) 4차원과 맞지 않아서 에러 뜸
# shape도 다르고 차원도 다른 상태. 차원부터 맞춰주자. N은 의미없음.


# model.add(Conv2D(128, (3,3), strides=1, input_shape=(28,28,1)))
# model.add(Conv2D(filters=100, kernel_size=(3,3), activation='relu'))
# model.add(Flatten())
# model.add(Reshape(target_shape=(28, 228)))

# model.add(Dense(100, input_shape=(28, 28)))

# model.add(LSTM(100, input_shape=(28, 28), activation='relu'))  

# model.summary()
# exit()


# 1. Conv2D 입력 (28, 28, 1)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))  # 출력: (28, 28, 32)

# 2. LSTM 입력을 위해 reshape: (28, 28*32)
model.add(Reshape((28, 28 * 32)))  # 출력: (28, 896)

# 3. LSTM 처리
model.add(LSTM(64, return_sequences=True))  # 출력: (28, 64)

# 4. 다시 Conv2D 입력 위해 reshape: (28, 8, 8) 또는 원하는 구조
model.add(Reshape((28, 8, 8)))  # 총 노드 수 유지 (28*64 = 1792 → 28*8*8=1792)

# 5. Conv2D 처리
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 출력: (28, 8, 64)

# 6. 다시 LSTM 위해 reshape: (28, 8*64)
model.add(Reshape((28, 8 * 64)))  # 출력: (28, 512)

# 7. LSTM
model.add(LSTM(128))  # 출력: (128,)


# model.add(LSTM(100, input_shape=(28, 28), activation='relu'))  


# model.add(Reshape(target_shape=(28, 10, 10)))
# model.add(Conv2D(128, (3,3), strides=1, input_shape=(28,28,1), activation='relu'))
# model.add(Flatten())
# model.add(Reshape(target_shape=(28, 10)))
# model.add(LSTM(100, input_shape=(28, 28), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=64, input_shape=(128,)))
model.add(Dense(units=10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
hist = model.fit(x_train, y_train, epochs=500, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

# 4. 평가, 예측
loss= model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

y_pred = model.predict(x_test)
print(y_pred)

# y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
y_test = y_test.values
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:' , round(acc, 4))
print('걸린시간:', end - start, '초')

# acc: 0.9915
# 걸린시간: 288.34585428237915 초