import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=76542
)

# 2. 모델구성
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
print("================================")
loss = model.evaluate(x_test, y_test)
results = model.predict([x])

print('loss:', loss)
print('[x]의 예측값:', results)

# 그래프 그리기
import matplotlib.pyplot as plt
plt.scatter(x, y)       # 데이터 점 찍기.
plt.plot(x, results, color='red')
plt.show()

# loss: 4.154255390167236
# [x]의 예측값: [[0.85975486]
#  [1.8483655 ]
#  [2.836976  ]
#  [3.8255866 ]
#  [4.8141966 ]
#  [5.8028073 ]
#  [6.7914176 ]
#  [7.780028  ]
#  [8.768639  ]
#  [9.757249  ]]



