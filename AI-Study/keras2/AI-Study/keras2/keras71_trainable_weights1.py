import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
# tf.random.set_seed(333) 
print(tf.__version__)   # 2.7.4

# 1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
'''
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
'''
print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32
# numpy=array([[ 0.13603318, -0.03480017,  0.7743634 ]], dtype=float32)>,
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32
#
# numpy= array([[-0.92561173,  0.8256177 ],
#        [ 0.6200088 ,  1.0182774 ],
#        [-0.5191052 , -0.6304303 ]], dtype=float32)>
# <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>
#
# <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32
# numpy= rray([[-0.02628279],
#        [-1.074922  ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

print("="*100)
print(model.trainable_weights)
print("="*100)

print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 6

#  전이학습 = 남이 잘 만들어놓은 모델을 내가 쓸거야. 날로먹는것
#  남이 만든 모델훈련을 안시키겠다 = 동결하겠다
#  속도가 매우 빠름. 이미 최상의 가중치이다 (huggingface에 수많은 가중치가 모여있음 가져다가 써!!!)
#  조건이 있음. 위 아래는 바꿔줘야한다. EXID 위아래. 전이학습 하면서 만든 노래이다. 캬~
#  위 아래 안바꾸면 성능이 떨어질 수 있다.

# ★★★★★ 동결하는 법 ★★★★★
model.trainable = False  # 훈련을 하지 않겠다. // 역전파 하지 않겠다. // 가중치 갱신을 하지 않겠다.
print(len(model.weights))               # 6
print(len(model.trainable_weights))     # 0

print("="*100)
print(model.weights)
print("="*100)
print(model.trainable_weights)          # []
print("="*100)

model.summary()
