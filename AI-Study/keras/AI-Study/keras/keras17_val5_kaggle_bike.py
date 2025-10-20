# https://www.kaggle.com/competitions/bike-sharing-demand

#  train.csv 와 new_test.csv로 count 예측

import numpy as np
import pandas as pd
# import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# print(sk.__version__)


# exit()
# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=361,
    )


# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=64,
          verbose=1,
          validation_split=0.1)

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
plt.title('캐글 바이크 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()


'''
# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])
y_predict = model.predict(x_test)

r2 = r2_score(y_test, results)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

y_submit = model.predict(new_test_csv)

print(submission_csv)
submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path + 'sampleSubmission_0522_1758.csv', index=False )

print('[loss]:', loss)
print('[rmse]:', rmse)
print('[r2]:', r2)
'''
# 0522_1758   rs 361  ep 100 bs 32
# [loss]: 0.17205165326595306
# [rmse]: 0.4147900345902945
# [r2]: 0.9999948256322875


# [loss]: 3.4662692546844482
# [rmse]: 1.8617921197320282
# [r2]: 0.9998957531271311


# validation_split 사용 후
# [loss]: 0.02154805324971676
# [rmse]: 0.14679233901275585
# [r2]: 0.9999993519516012