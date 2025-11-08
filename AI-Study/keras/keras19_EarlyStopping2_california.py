# 17_2 copy
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk
print(sk.__version__)   # 1.1.3
import tensorflow as tf
print(tf.__version__)   # 2.9.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

# 1. 데이터
dataset = fetch_california_housing()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape)  # (20640, 8)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                        test_size=0.1,
                        random_state=195) #21 , 6, 36
                                                  

# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',                    # 최대값 max, 알아서 찾아줘: auto
    patience=20,                     # 이만큼 참을 거다. local minimal, global minimal
    restore_best_weights=False,       #EarlyStopping 의 Default는 False. 최적의 weight 때 멈춘다. 최소지점 save.
)

hist = model.fit(x_train, y_train, epochs=400, batch_size=32,
          verbose=1,
          validation_split=0.3,
          callbacks=[es],
          )


print('=========  hist  ========')
print(hist)     # <keras.callbacks.History object at 0x0000022D52E9AC10>
print('=========  history  ========')
print(hist.history)
print('=========  loss  ========') #loss 값만 보고 싶을 경우.
print(hist.history['loss'])
print('=========  val_loss  ========')
print(hist.history['val_loss'])

# 그래프 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')  # y값만 넣으면 시간순으로 그린 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()

# 4. 평가, 예측
print('=====================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test]) # 원래의 y값과 예측된 y값의 비교
print('loss:', loss)
# print('[x]의 예측값:', results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어:', r2)   

# loss: 0.6058887839317322
# r2 스코어: 0.5458594509681596          #  기준 : R2 > 0.59

# validation_split 사용 후
# 


# restore_best_weights=True
# loss: 0.4720142185688019
# r2 스코어: 0.653795740279342
# val_loss: 0.44698432087898254

# restore_best_weights=False  Epoch 132/400 Patience 20
# loss: 0.4305768311023712
# r2 스코어: 0.6841885259934934
# val_loss: 0.42820867896080017