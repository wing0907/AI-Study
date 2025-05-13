import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
             [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60] ])        # (13, 3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape)        # ((13, 3) (13,)


x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)                 # (13, 3, 1)  (batch_size, timesteps, feature)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=190,                   
    shuffle=True, 
)

#  2. 모델구성
model = Sequential()
model.add(LSTM(units=64, input_length=3, input_dim=1, activation='relu')) #  return_sequences=True,
# model.add(GRU(units=10, input_shape=(3, 1), activation='relu'))
# model.add(SimpleRNN(units=10, input_shape=(3, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))


#  3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=2000)

model.compile(loss= 'mse', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='auto',
                   patience=30, verbose=2,
                   restore_best_weights=True,
                   )

model.fit(x_train, y_train, epochs=1000, batch_size=12,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es],
                 )

#  4. 평가, 예측
# results = model.evaluate(x, y)
# print('loss : ', results)


loss = model.evaluate(x_test, y_test, verbose=2)
print('loss : ', loss[0])



# 예측함
x_pred = np.array([50,60,70]).reshape(1,3,1)  # (3,) -> (1, 3, 1)
y_pred = model.predict(x_pred)
print('[50,60,70]의 결과 : ', y_pred)           # 80 맞추기

# loss :  [0.5647363066673279, 0.0]
# [50,60,70]의 결과 :  [[81.2901]]

# loss :  0.07004515081644058
# [50,60,70]의 결과 :  [[79.59919]]