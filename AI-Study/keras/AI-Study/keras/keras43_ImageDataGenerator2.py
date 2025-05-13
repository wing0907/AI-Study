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

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)     # (160, 200, 200, 1) (160,)
print(x_test.shape, y_test.shape)       # (120, 200, 200, 1) (120,)


model = Sequential()
model.add(Conv2D(32, (2,2), strides=1, input_shape=(200,200,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath=filepath
# )


start = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=4,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],)
end = time.time()

# 평가
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])

# 예측
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).flatten()  # (N, 1) → (N,)
y_test = y_test.flatten()                      # (N, 1) → (N,)

# 정확도
acc = accuracy_score(y_test, y_pred)
print('acc:', round(acc, 4))
print('걸린시간:', end - start, '초')

# 시각화용 reshape
images = x_test.reshape(-1, 200, 200)

# 시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(16, 6))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss :  0.0348055437207222
# acc :  0.9916666746139526
# acc: 0.9917
# 걸린시간: 149.21810269355774 초


# loss :  0.02179562672972679
# acc :  1.0
# acc: 1.0
