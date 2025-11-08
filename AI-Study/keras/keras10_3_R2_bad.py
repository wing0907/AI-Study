# 20250611 보강
#1. R2를 음수가 아닌 0.5 이하로 만들것
#2. 데이터 건들지 말 것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 사이즈 75%
#7. epoch 100번 이상
#8. loss지표는 mse
# [실습시작]


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.metrics import AUC

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9,6, 17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=45
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(
    loss='mse',
    optimizer='adam',
)
model.fit(x_train, y_train, epochs=101, batch_size=1)

# 4. 평가, 예측
print("==================================")
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)



from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)    

r2 = r2_score(y_test, results)
print('loss:', loss)
print('r2 스코어 :', r2)
print('RMSE:', rmse)
print('[x]의 예측값:', results)

# loss: 46.014976501464844
# r2 스코어 : 0.2881345829657588
# RMSE: 6.783434274546585
# [x]의 예측값: [[ 2.3921669]
#  [13.810293 ]
#  [ 8.401707 ]
#  [12.608387 ]
#  [13.20934  ]]

# loss: 23.733957290649414
# r2 스코어 : 0.3052120710400703
# RMSE: 4.871750779059947
# [x]의 예측값: [[11.6041   ]
#  [11.664551 ]
#  [11.66975  ]
#  [ 3.1632845]
#  [11.660127 ]]


