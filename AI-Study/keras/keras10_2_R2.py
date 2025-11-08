import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8, 14,15, 9,6, 17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=76542
)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(400))
model.add(Dense(50))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=4)

# 4. 평가, 예측
print("==================================")
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

print('loss:', loss)
print('[x]의 예측값:', results)

from sklearn.metrics import r2_score, mean_squared_error

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))

# rmse = RMSE(y_test, results)    
# print('RMSE:', rmse)

r2 = r2_score(y_test, results)
print('r2 스코어 :', r2)

# loss: 19.63554573059082
# [x]의 예측값: [[ 9.399217 ]
#  [ 6.327688 ]
#  [15.542274 ]
#  [ 4.2800026]
#  [10.423059 ]
#  [18.613804 ]]
# r2 스코어 : 0.3069807291030884




