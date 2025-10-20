import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional # Bidirectional 자체가 모델이 아니다. 단지 양방향으로 감쌀거야(wrap)란 뜻.

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])


x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],])        # (7, 3)
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)        # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)                 # (7, 3, 1)  (batch_size, timesteps, feature)

# 2. 모델구성
model = Sequential()
# model.add(GRU(units=10, input_shape=(3, 1)))
model.add(Bidirectional(SimpleRNN(units=10), input_shape=(3, 1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 565

############## param 개수 ##################
"""
RNN :  120
Bidirectional :  240


GRU :  390
Bidirectional :  780


LSTM : 480
Bidirectional : 960
"""