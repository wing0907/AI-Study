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
model.add(Dense(10, input_dim=1))
model.add(Dense(200))
model.add(Dense(400))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=3)

# 4. 평가, 예측
print("==================================")
loss = model.evaluate(x_test, y_test)
results = model.predict([x])

print('loss:', loss)
print('[x]의 예측값:', results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y, c='red') # 데이터 점 찍기.
plt.plot(x, results, color='black')
plt.show()

# loss: 19.551843643188477
# [x]의 예측값: [[ 0.14194609]
#  [ 1.1034905 ]
#  [ 2.065035  ]
#  [ 3.0265796 ]
#  [ 3.9881237 ]
#  [ 4.9496684 ]
#  [ 5.9112124 ]
#  [ 6.8727565 ]
#  [ 7.8343015 ]
#  [ 8.795845  ]
#  [ 9.757391  ]
#  [10.718937  ]
#  [11.680481  ]
#  [12.642025  ]
#  [13.603569  ]
#  [14.565113  ]
#  [15.526658  ]
#  [16.488201  ]
#  [17.449747  ]
#  [18.411291  ]]

