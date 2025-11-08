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
# vgg16.trainable=True       # 가중치 안동결
                           # 우리반에 태영이가 있고 태영이가 아닐 시 안태영...이란다.. 헣헣

model = Sequential()
model.add(vgg16)
model.add(Flatten())  # False하고 요로케 flatten 붙이면 됨
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

print(len(model.weights))  # 30  =  (maxpool 3개 빼면 vgg16 에서 -3 => 13 + 13 = 26에다가 Dense 2 + bias 2 => 4
print(len(model.trainable_weights))  # 4

# trainable = True  # 30, 30
# trainable = False # 30, 4

import pandas as pd
pd.set_option('max_colwidth', None) # None 길이 다나옴, 10 = 10개만 나옴
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
#                                                           Layer Type Layer Name  Layer Trainable
# 0  <keras.engine.functional.Functional object at 0x00000265B05A52E0>      vgg16            False
# 1   <keras.layers.core.flatten.Flatten object at 0x00000265B05B37C0>    flatten             True
# 2       <keras.layers.core.dense.Dense object at 0x00000265B6C0DF40>      dense             True
# 3       <keras.layers.core.dense.Dense object at 0x00000265B6C462B0>    dense_1             True
# pd.set_option('max_colwidth', None) = ↑↑↑↑↑요부분 길이↑↑↑↑↑
