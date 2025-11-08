from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)      # (50000, 10) (10000, 10)


model = Sequential()
model.add(Conv2D(128, (2,2), strides=1, input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())                                               # MaxPooling 한방 때리면 반띵됨
model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D())                                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
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
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")   

path = './_save/keras39/'
filename = '.hdf5'
filepath = "".join([path, 'k39_03',filename])
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

# loss :  1.0032650232315063
# acc: 0.7549
# 걸린시간: 423.5881996154785 초

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
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()