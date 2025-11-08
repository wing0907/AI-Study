import sklearn as sk
print(sk.__version__)   # 1.6.1 -> 1.1.3
 
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(506, 13)
print(y)
print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    random_state=97
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(200))
model.add(Dense(600))
model.add(Dense(400))
model.add(Dense(50))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=4)

# 4. 평가, 예측
print('========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss:', loss)
print('[x]의 예측값:', results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어:', r2)     #0.75 이상


# r2 스코어: 0.7630468715814822