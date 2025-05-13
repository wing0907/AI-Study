from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
x_train = np.load(np_path + "keras46_05_x_train.npy")
y_train = np.load(np_path + "keras46_05_y_train.npy")

end = time.time()


datagen = ImageDataGenerator(
    # rescale=1./255,           #정규화 0~255 scaling.
    horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환 /  좌우반전
    # vertical_flip=True,       #수직반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,    #수치를 10% 이동 하겠다. 평행이동 10%
    # height_shift_range=0.1, #수직이동 10%
    rotation_range=15,      #360도 중 n도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
)
# print(x_train[0].shape)

augment_size = 200        # 6만개 -> 10만개로 만들거야.

randidx = np.random.randint(x_train.shape[0], size=augment_size)  # x_train의 60000의 사이즈는 40000이다
          # np.random.randint(60000, 40000)
print(randidx)                              # [17528 27056 39564 ... 52637 32955 52229]
print(np.min(randidx), np.max(randidx))     # 1 59998

x_augmented = x_train[randidx].copy()       # 4만개의 데이터 copy, copy로 새로운 메모리 할당.
                                            # 서로 영향 x

y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape)                    # (40000, 28, 28)
print(y_augmented.shape)                    # (40000,)

# x_augmented = x_augmented.reshape(40000, 28, 28, 1)
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3, )
print(x_augmented.shape)                    # (40000, 28, 28, 1)

x_augmented = datagen.flow(
    x_augmented,
    y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]

print(x_augmented.shape)                    # (40000, 28, 28, 1)

print(x_train.shape)                        # (2048, 100, 100, 3)
x_train = x_train.reshape(-1, 100, 100, 3)
# x_test = x_test.reshape(-1, 100, 100, 3)
# print(x_train.shape, x_test.shape)          # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))    # shape 때문에 2개 맞추는 거임
print(x_train.shape, y_train.shape)         # (2248, 100, 100, 3) (2248, 3)


print(np.max(x_train), np.min(x_train)) # 1.0 0.011764707
# print(np.max(x_test), np.min(x_test))   # 1.0  0.0


x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=333,                   
    shuffle=True, 
)
path = './_save/rps/'
model = load_model(path + 'k46_06.hdf5')



# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=40, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x_train, y_train, epochs=500, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],
                 )

# 4. 평가, 예측
loss= model.evaluate(x1_test, y1_test, verbose=1)
print('loss : ', loss[0])
print('acc:' , round(loss[1], 4))

y_pred = model.predict(x1_test)


# # y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
# y_test = y_test.values
# y_test = np.argmax(y_test, axis=1)
# y_pred = np.argmax(y_pred, axis=1)



# print('걸린시간:', end - start, '초')

# acc: 0.9028
# 걸린시간: 267.4948709011078 초


# 예측값과 정답값 준비 (예시용)
y_pred = model.predict(x1_test)
y_pred = (y_pred > 0.5).astype(int)  # 분류 모델일 경우

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x1_test.reshape(-1, 100, 100, 3)  # x_test가 (N, 64)인 경우만 OK

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='winter')
    plt.title(f"예측:{y_pred[i]} / 정답:{y1_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# loss :  0.2756417691707611
# acc: 0.9615
