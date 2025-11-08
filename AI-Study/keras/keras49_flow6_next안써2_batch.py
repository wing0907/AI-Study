from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

augment_size = 100  # 증가시킬 사이즈

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)        # (60000, 28, 28)
print(x_train[0].shape)     # (28, 28)  0~59999까지.

# plt.imshow(x_train[0], cmap='gray')
# plt.show()

aaa = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1)
print(aaa.shape)


datagen = ImageDataGenerator(
    rescale=1./255,           #정규화 0~255 scaling.
    # horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환 /  좌우반전
    vertical_flip=True,       #수직반전 <- 데이터 증폭 또는 변환 / 상하반전
    width_shift_range=0.1,    #수치를 10% 이동 하겠다. 평행이동 10%
    # height_shift_range=0.1, #수직이동 10%
    # rotation_range=15,      #360도 중 n도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
)


xy_data = datagen.flow(
    aaa,                      # x데이터
    np.zeros(augment_size), # y데이터 생성, 전부 0으로 가득찬 y값.
    batch_size=32,
    shuffle=False,      
)#.next()                     # 요 데이터의 첫번째거 // 1차원으로 만든다음 증폭하는 것이 가장 안전하다.
                             ## .next 했을 경우 tuple 상태, iterator의 첫번째거만 출력한다.
# x y 를 넣으면 tuple 형태로 만들어주고 y 를 빼면 numpy.ndarray 형태로 만들어 줌.

print(xy_data)
print(type(xy_data))          # <class 'keras.preprocessing.image.NumpyArrayIterator'>
print(len(xy_data))           # 4

print(xy_data[0][0].shape)       # (32, 28, 28, 1) 
print(xy_data[0][1].shape)       # (32,)

print(xy_data[3][0].shape)       # (4, 28, 28, 1) 
print(xy_data[3][1].shape)       # (4,)


plt.figure(figsize=(7, 7))
for i in range(32):
    plt.subplot(7, 7, i+1)
    plt.imshow(xy_data[0][0][i], cmap='gray') # .next빼면 차원하나 더 늘려준다. # x데이터의 0번째의 첫번째부터. 통배치니깐 결국엔 전체표기

plt.show()

