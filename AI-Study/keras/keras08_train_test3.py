from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y= np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,)
# print(y.shape)  # (10,)

# [실습] train과 test를 섞어서 sklearn으로 7:3으로 나눠라.

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) blog

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                            # train_size=0.7, # 디폴트: 0.75
                            test_size=0.3, # 생략가능, 디폴트: 0.25
                            shuffle=True,    # 디폴트: True
                            random_state=764821,
                            )
print(x_train, x_test)  #[1 2 3 4 5 6 7] [ 8  9 10] shuffle=False 한 데이터 값
print(y_train, y_test)  #[4 7 2 5 1 6 3] [ 9 10  8] shuffle=True 한 데이터 값 (디폴트)


# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3)

# 4.  평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss:', loss)
print('[11]의 예측값:', results)

# loss: 7.221512419164355e-07
# [11]의 예측값: [[10.998702]]

