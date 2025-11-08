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
# model.add(SimpleRNN(64, input_shape=(3, 1), activation='relu'))
# model.add(SimpleRNN(units=64, input_shape=(3, 1), activation='relu'))

model.add(SimpleRNN(units=10, input_length=3, input_dim=1))
# input_length = timesteps, input_dim = feature

# model.add(SimpleRNN(units=10, input_dim=1, input_length=3,))
# 순서가 바뀌어도 잘 돌아간다. 대신 가독성이 떨어짐


model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))



#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#  4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)
# print("acc : ", round(results, 4)) 

# 예측함
x_pred = np.array([8,9,10]).reshape(1,3,1)  # (3,) -> (1, 3, 1)
y_pred = model.predict(x_pred)
print('[8,9,10]의 결과 : ', y_pred)


