import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()       # x만 준비하고 y는 자리만 명시해주기

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
# DNN 하기 위해 60000, 28*28 로 reshape하고 255.으로 나눈건 scaling 완료
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

# 2. 모델
input_img = Input(shape=(28*28,))

###### 인코더
# encoded = Dense(1, activation='relu')(input_img) # 엄청 흐림
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(64, activation='tanh')(input_img)  # tanh + sigmoid 조합 나쁘지 않음
# encoded = Dense(1024, activation='relu')(input_img) # 증폭해서 잘보임

###### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)       # linear랑 비슷함
decoded = Dense(28*28, activation='sigmoid')(encoded)    #64로 해도 훨씬 깔끔하게 나옴 
# decoded = Dense(28*28, activation='tanh')(encoded)         # 개판 뜸

autoencoder = Model(input_img, decoded)

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') # mse 보다 잘 보임. 결과치 보고 성능 판단ㄱㄱ
autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.2)

# 4. 평가, 예측
decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

