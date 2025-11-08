import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#  1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

#  [실습] train_test_split 10:3:3 으로 나눈다.

# x_train =  x[:10]
# y_train = y[:10]

# x_val =  x[10:14]
# y_val = y[10:14]

# x_test = x[14:]
# y_test = y[14:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85,
    shuffle=True,
    random_state=42,
)
print(x_train)      # [15 14 12  9 10  3 16  5  8 11 13  4  7]
print(x_test)       # [1 2 6]
print(y_train)
print(y_test)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8,
    shuffle=True,
    random_state=42,
)

print(x_train)      # [ 8  3 12 14  7 10  5 13  9 16]
print(x_val)       # [ 4 11 15]



exit()


"""
##################################################################################
x_1 = x[0:14]
print('x_train', x_1)
x_val = x[14:]
print('x_val', x_val)
x_test = x[14:]
print('x_test', x_test)

y_1 = y[0:14]
y_val = y[14:]
y_test = y[14:]

x_train, x_test, y_train, y_test = train_test_split(x_1,y_1,test_size=0.2,random_state=44,shuffle=False)

# x_train1, x_test1, x_val1, y_val1 = train_test_split(x_train,y_train, test_size=0.2, shuffle=False,
#     x, y,
#     test_size=10,
#     random_state=15,
# )
###################################################################################################
"""
# exit()
#  2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=1, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))



#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_data=(x_val, y_val))


#  4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

results = model.predict([17])
print('[17]의 예측값 : ', results)
