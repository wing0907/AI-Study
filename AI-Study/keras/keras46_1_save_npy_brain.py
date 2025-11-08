import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 인스턴스를 정의한다 클래스를 만들어서 변수를 정의한다. 고가시가 인스턴스화 한다.
# 이미지를 가져다가 직접 수치화 하는 함수쓰
train_datagen = ImageDataGenerator(
    rescale=1./255,         #정규화 0~255 scaling.
    # horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환
    # vertical_flip=True,     #수직반전 <- 데이터 증폭 또는 변환
    # width_shift_range=0.1,  #수치를 10% 이동 하겠다. 평행이동 10%
    # height_shift_range=0.1, #수직이동 10%
    # rotation_range=5,       #360도 중 5도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    # fill_mode='nearest'     
    
)

test_datagen = ImageDataGenerator(      #test에는 증폭 또는 변환 시키지 않는다. 평가 데이터는 수정하지 않는다.
    rescale=1./255,
)

path_train = 'C:\Study25\_data\image\\brain\\train\\'
path_test = 'C:\Study25\_data\image\\brain\\test\\'
start = time.time()
xy_train = train_datagen.flow_from_directory(
    path_train,                    #경로
    target_size=(200, 200),        #원 사이즈 150,150 가 해당 사이즈로 자동변환 됨 resize
    batch_size=160,                #메모리 사이즈 때문에 batch를 통배치로 하지 않는다
    class_mode='binary',           #이진분류
    color_mode='grayscale',        #색깔
    shuffle=True,
    seed=333,
)
# Found 160 images belonging to 2 classes. 

xy_test = test_datagen.flow_from_directory(
    path_test,                     #경로
    target_size=(200, 200),        #원 사이즈 150,150 가 해당 사이즈로 자동변환 됨 resize
    batch_size=120,                 #
    class_mode='binary',           #이진분류
    color_mode='grayscale',        #색깔
    # shuffle=True, 평가라서 shuffle 할 필요없음
)
# Found 120 images belonging to 2 classes.
end = time.time()
print('time:', round(end- start, 2), "seconds")
# time: 0.02 seconds
all_x = []
all_y = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)


all_x1 = []
all_y1 = []

for i in range(len(xy_test)):
    x1_batch, y1_batch = xy_test[i]
    all_x1.append(x1_batch)
    all_y1.append(y1_batch)


x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)

x1 = np.concatenate(all_x1, axis=0)
y1 = np.concatenate(all_y1, axis=0)

print('x.shape :', x.shape) # x.shape : (160, 200, 200, 1)
print('y.shape :', y.shape) # y.shape : (160,)

print('x1.shape :', x1.shape) # x1.shape : (120, 200, 200, 1)
print('y1.shape :', y1.shape) # y1.shape : (120,)
end2 = time.time()
print('save time:', round(end2- end, 2), "seconds") 
# save time: 0.19 seconds
start2 = time.time()
np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + "keras46_02_x_train.npy", arr=x)
np.save(np_path + "keras46_02_y_train.npy", arr=y)
np.save(np_path + "keras46_02_x_test.npy", arr=x1)
np.save(np_path + "keras46_02_y_test.npy", arr=y1)
end3 = time.time()          # npy save time: 374.3 seconds 
print('npy save time:', round(end3- start2, 2), "seconds") 
# npy save time: 0.05 seconds
