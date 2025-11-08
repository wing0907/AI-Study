import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import tensorflow as tf
import random
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from tensorflow.keras.applications import DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, GlobalAvgPool2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    input_shape=(100,100,3),
)

np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + "keras44_01_x_train.npy", arr=x)
# np.save(np_path + "keras44_01_y_train.npy", arr=y)

start = time.time()
x1_train = np.load(np_path + "keras46_03_x_train.npy")
y1_train = np.load(np_path + "keras46_03_y_train.npy")
# x_test = np.load(np_path + "keras46_02_x_test.npy")
# y_test = np.load(np_path + "keras46_02_y_test.npy")
end = time.time()

print(x1_train)
print(y1_train[:20])    # [1. 0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1. 1. 0. 0. 1.]
print(x1_train.shape, y1_train.shape)  # (1027, 100, 100, 3) (1027,)
print("load time :", round(end-start, 2),"seconds")
#  load time : 33.87 seconds

# x1_train = x1_train.reshape(-1, 100*100, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x1_train, y1_train, test_size=0.3, random_state=SEED,                   
    shuffle=True, 
)


# vgg16.trainable=False       # 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())  # False하고 요로케 flatten 붙이면 됨
model.add(GlobalAvgPool2D())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
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
# filepath = "".join([path, 'k39_0444',filename])
# #######################################################
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath=filepath
# )

start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=256, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.1,
                 callbacks=[es,],)
end = time.time()

# 4. 평가, 예측
loss= model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

# y_pred = model.predict(x_test)
# print(y_pred)

# y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
# y_test = y_test.values
# y_test = np.argmax(y_test, axis=1)   # 다중일 경우 사용
# y_pred = np.argmax(y_pred, axis=1)

y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten() # flatten converts [[1],[0]] to [1,0]

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


# ================ [기존 결과값 (epoch최소100이상)] ================
# loss :  0.0033516467083245516
# acc :  1.0
# 걸린시간: 80.49675488471985 초
# -----------------------------------------------------------------



# ================ [가중치 동결 + Flatten , epochs=5] ==============
# loss:  0.0769
# acc: 0.9838
# 걸린시간: 24.03498935699463 초
# ------------------------------------------------------------------



# ================ [가중치 비동결 + Faltten , epochs=5] ==============
# loss:  32.3747
# acc: 0.4887
# 걸린시간: 37.381181955337524 초
# -------------------------------------------------------------------



# ================ [가중치 동결 + GAP , epochs=5] ================
# loss:  0.4097
# acc: 0.7767
# 걸린시간: 23.743753671646118 초
# ---------------------------------------------------------------



# ================ [가중치 비동결 + GAP , epochs=5] ===============
# loss:  66.7572
# acc: 0.4887
# 걸린시간: 37.23177623748779 초
# ----------------------------------------------------------------

