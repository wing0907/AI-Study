import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.decomposition import PCA

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()       # x만 준비하고 y는 자리만 명시해주기

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
# DNN 하기 위해 60000, 28*28 로 reshape하고 255.으로 나눈건 scaling 완료
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 평균 0, 표편 0.1인 정규분포 형태의 랜덤값)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape) # 옛날사진 복원하는 작업이라고 생각하면 됨
    
print(x_train_noised.shape, x_test_noised.shape)    # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))              # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised)) # 1.4941472990656228 -0.5302963833813626 생각보다 넓지만 그리 넓은게 아니다

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)
print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0


# 2. 모델
input_img = Input(shape=(28*28,))

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

# 0.95 이상 : 154
# 0.99 이상 : 331
# 0.999 이상 : 486
# 1.0 일때 : 784


hidden_size = 154

model = autoencoder(hidden_layer_size=hidden_size)

# 3. 컴파일, 훈련
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss='binary_crossentropy') # mse 보다 잘 보임. 결과치 보고 성능 판단ㄱㄱ
model.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20,7))
    
# 이미지 5개 랜덤
random_images = random.sample(range(decoded_imgs.shape[0]), 5)

#  원본 이미지 맨위에 그리기
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
