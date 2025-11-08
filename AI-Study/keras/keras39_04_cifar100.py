from tensorflow.keras.datasets import cifar100
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#  1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



#  1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000, )

print(np.unique(y_train, return_counts=True))

print(pd.value_counts(y_test))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

######### 스케일링 2. 정규화 (많이 사용함) 데이터를 0에서 1로 만드는 것
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0  0.0
print(np.max(x_test), np.min(x_test))   # 1.0  0.0

print(x_train.shape, x_test.shape)      # (50000, 32, 32, 3) (10000, 32, 32, 3)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (50000, 100) (10000, 100)


model = Sequential()
model.add(Conv2D(512, (3,3), strides=1, input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())      
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())                                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(1024, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=100, activation='softmax'))
model.summary()




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

path = './_save/keras39/'
filename = '.hdf5'
filepath = "".join([path, 'k39_04',filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=256, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.1,
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

# loss :  2.2111172676086426
# acc :  0.44830000400543213
# 걸린시간: 1398.0711462497711 초

# aaa = 3
# print(y_train[aaa])

# import matplotlib.pyplot as plt
# # plt.imshow(x_train[1], 'gray')
# plt.imshow(x_train[aaa])
# plt.show()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x_test.reshape(-1, 32, 32, 3)  # x_test가 (N, 64)인 경우만 OK

plt.figure(figsize=(12, 4))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(images[i])
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()