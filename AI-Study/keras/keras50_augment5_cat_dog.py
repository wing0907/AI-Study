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


x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
# x1_test = np.load(np_path + "keras44_01_x_test.npy")
# y1_test = np.load(np_path + "keras44_01_y_test.npy")


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

augment_size = 1000        

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

print(x_train.shape)                        # (60000, 28, 28)
x_train = x_train.reshape(-1, 50, 50, 3)
# x_test = x_test.reshape(-1, 50, 50, 3)
# print(x_train.shape, x_test.shape)          # (25000, 50, 50, 3) (12500, 50, 50, 3)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))    # shape 때문에 2개 맞추는 거임
print(x_train.shape, y_train.shape)         # (26000, 50, 50, 3) (26000,)


print(np.max(x_train), np.min(x_train)) # 1.0  0.0
# print(np.max(x_test), np.min(x_test))   # 1.0  0.0

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_train, y_train, test_size=0.1, random_state=190, stratify=y_train,                  
    shuffle=True, 
)


path = './_save/cat_dog/'
model = load_model(path + 'k45_99.hdf5')

# 3. 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=40, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x1_train, y1_train, epochs=500, batch_size=512, # batch는 행이다!!!!!!!!
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],
                 )

# 4. 평가, 예측
loss = model.evaluate(x1_test, y1_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])


# 예측
y_pred = model.predict(x1_test)

y_submit = model.predict(x1_test) 

# # 시각화용 reshape
images = x1_test.reshape(-1, 50, 50, 3)

# 시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(16, 6))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i], cmap='winter')
    plt.title(f"예측:{y_pred[i]} / 정답:{y1_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss :  0.40358683466911316
# acc :  0.8123077154159546

