from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

augment_size = 100  # 증가시킬 사이즈

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

aaa = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)


datagen = ImageDataGenerator(
    rescale=1./255,         #정규화 0~255 scaling.
    # horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환 /  좌우반전
    vertical_flip=True,     #수직반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,  #수치를 10% 이동 하겠다. 평행이동 10%
    # height_shift_range=0.1, #수직이동 10%
    # rotation_range=15,       #360도 중 n도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
)


xy_data = datagen.flow(
    aaa,                        # x데이터
    np.zeros(augment_size),     # y데이터 생성, 전부 0으로 가득찬 y값.
    batch_size=augment_size,
    shuffle=False,      
).next()                        # 요 데이터의 첫번째거 // 1차원으로 만든다음 증폭하는 것이 가장 안전하다.



print(xy_data)
print(type(xy_data))            # <class 'tuple'>
print(len(xy_data))             # 2 // .next빼면 1이 나옴.

print(xy_data[0].shape)         # data x (100, 28, 28, 1)
print(xy_data[1].shape)         # data y (100,) 1번째는 np.zeros(augment_size) 요녀석이 된다.

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()

