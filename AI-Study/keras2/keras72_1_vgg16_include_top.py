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

######## default #########
model = VGG16(weights='imagenet',
              include_top=False,     # 우리의 선조는 이미 알고 있어서 노래도 만들었다. 위 아래 위위 아래..EXID...하..cifa..r..
              input_shape=(100, 100, 3),  # 최소 (32, 32, 3) 고로 mnist는 안됨 그치만 시키신다고 함 헣헣
              )                           # (224, 224, 3) = imagenet에 사용된 VGG16의 default shape.
                                          
##########################

# model = VGG16()  # 여기에 모델 하나씩 넣어서 다운받기

model.summary()

# VGG16   (include_top=True)
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0              # padding 은 same = default

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0

#  flatten (Flatten)           (None, 25088)             0
#  fc1 (Dense)                 (None, 4096)              102764544
#  fc2 (Dense)                 (None, 4096)              16781312
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0


# VGG16   (include_top=False)  Flatten을 날림 그래서 총 파라미터 수가 적어짐
# Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0
#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#  block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928
#  block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0

#  block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856
#  block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584
#  block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0

#  block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168
#  block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080
#  block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080
#  block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0

#  block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160
#  block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808
#  block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808
#  block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0

#  block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0

########### include_top=False ###########
# 1. input_shape를 우리가 훈련 시킬 데이터의 shape로 수정
# 2. FC layer 없어짐 (직접 아래에 fc layer 붙여주면 됨.)
# FC는 football club 아니다. fully connected 이다! FC에 환장한놈이 매기수에 있단다.. 난가..홓홓

