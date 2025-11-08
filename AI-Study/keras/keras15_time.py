#  14 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time # 시간에 대한 모듈 import

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y= np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,)
# print(y.shape)  # (10,)

x_train = np.array(range(100))
y_train = np.array(range(100))

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()    # 현재 시간을 반환, 시작시간.
print(start_time)           # 1747968510.8221679  특정 기준 시간에서 이만큼 지난 시간
# exit()
model.fit(x_train, y_train, epochs=1000, batch_size=128,
          verbose=0
          )            
# 0 = 침묵. 빨리 넘기기
# 1 = default.
# 2 = 프로그래스바 삭제. 간결해짐
# 3 = 에포만 나옴. epoch 만 확인하고 싶으면 0,1,2 이외의 숫자 입력
end_time = time.time()      
print("걸린시간 : ", end_time - start_time, '초')

"""
# 1. 1000에포에서 0, 1, 2, 3의 시간을 적는다.
0 : 40.823861837387085 초
1 : 53.012932538986206 초
2 : 42.52881479263306 초
3 : 41.88542461395264 초

# 2. 1000에포에서 verbose = 1의 시간을 적는다.
# batch 1, 32, 128 일때 시간.
1 : 52.60656929016113 초
32 : 3.958158254623413 초
128 : 1.9499473571777344 초
"""

