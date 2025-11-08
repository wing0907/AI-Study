import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# VGG16 이라는 모델이 있다 그거슨 2014년 이미지넷 대회에서 준우승한 16층 CNN 모델이다.
# 단순히 CNN 레이어가 16개여서 VGG16 임
# 다운이 안되면 cmd 에서 가상환경 들어가서 -> conda install openssl  입력!

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169
from tensorflow.keras.applications import DenseNet201, InceptionV3, InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D

vgg16 = VGG16(
    include_top=False,
    input_shape=(32,32,3),
)

vgg16.trainable=False       # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())  # False하고 요로케 flatten 붙이면 됨
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

### 실습 ###
# 비교할 것
# 1. 이전의 본인이 한 최상의 결과가
# 2. 가중치를 동결하지 않고 훈련시켰을 때, trainable=True
# 3. 가중치를 동결하고 훈련시켰을 때, trainable=False
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
