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
    input_shape=(50,50,3),
)

path = 'C:/Study25/_data/kaggle/cat_dog/'
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
x_test = np.load(np_path + "keras44_01_x_test.npy")
y_test = np.load(np_path + "keras44_01_y_test.npy")
end = time.time()

print("x_train shape:", x_train.shape)
print("y_train[:20]:", y_train[:20])
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("load time:", round(end - start, 2), "seconds")

# reshape for LSTM: (samples, timesteps, features)
x_train = x_train.reshape(25000, -1)  # (25000, 120000)
x_test = x_test.reshape(12500, -1)    # (12500, 120000)


print(x_train.shape, x_test.shape)

# reshape to (timesteps=400, features=300)
x_train = x_train.reshape(-1, 50,50, 3)
x_test = x_test.reshape(-1, 50,50, 3)

x11_train, x11_test, y11_train, y11_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=SEED, shuffle=True)

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
# loss :  0.6352799534797668
# acc :  0.6469333171844482
# 걸린시간: 10.798173904418945 초
# -----------------------------------------------------------------



# ================ [가중치 동결 + Flatten , epochs=5] ==============
# loss:  1.4411
# acc: 0.4
# 걸린시간: 38.66892147064209 초
# ------------------------------------------------------------------



# ================ [가중치 비동결 + Faltten , epochs=5] ==============
# loss:  4.7151
# acc: 0.18
# 걸린시간: 95.02459979057312 초
# -------------------------------------------------------------------



# ================ [가중치 동결 + GAP , epochs=5] ================
# loss:  1.4412
# acc: 0.4
# 걸린시간: 38.222686767578125 초
# ---------------------------------------------------------------



# ================ [가중치 비동결 + GAP , epochs=5] ===============
# loss:  4.8703
# acc: 0.06
# 걸린시간: 95.60718083381653 초
# ----------------------------------------------------------------

