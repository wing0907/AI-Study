import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# SimpleRNN = 간단한 RNN

# 1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열 데이터가 될 수 있다. (온도, 월급 등 시간순서대로 이어져있다고 볼 수 있음)
# 시계열 데이터가 3차원 이지만, 2차원 데이터로 받을 수 있다. 그럴땐 timesteps와 features를 나누는 방식으로 reshape 해줘야 한다.
# 시계열 데이터는 y 값을 주지 않는다. 실무에선 datasets 처럼 데이터를 받게 된다. x와 y값을 n단위로 짤라서 분류하는건 나의 몫. 함수로 지정할 수 있음.
# 1,2,3 다음은 4 / 2,3,4 다음은 5로 학습시키는 것. 8, 9, 10 다음은 데이터가 없음으로 9까지 나누고 8,9,10 다음 11을 예측하는 설계를 함.

x = np.array([[1,2,3],
             [2,3,4],
             [3,4,5],
             [4,5,6],
             [5,6,7],
             [6,7,8],
             [7,8,9],])        # (7, 3)
y = np.array([4,5,6,7,8,9,10])
# ex) 주가 데이터, 내일의 주가를 예측해라

print(x.shape, y.shape)        # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)                 # (7, 3, 1)  (batch_size, timesteps, feature)
# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#              ...
#              [[7],[8],[9]],
#                ])



#  2. 모델구성
model = Sequential()
# # model.add(SimpleRNN(64, input_shape=(3, 1), activation='relu'))
# model.add(LSTM(units=10, input_shape=(3, 1), activation='relu')) #  return_sequences=True,
model.add(GRU(units=10, input_shape=(3, 1), activation='relu'))
# model.add(SimpleRNN(units=10, input_shape=(3, 1), activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0

# 파라미터 개수 = feature * units + units * units + bias * units
            # = (1 * 10) + (10 * 10) + (1 * 10) = 120
            # = units * (feature + units + bias)
            # = 10 * (1 +  10 + 1) = 120
# RNN의 단점은 timesteps가 길어지면 그 이전의 데이터를 기억하지 못한다.