# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd               # 전처리
print(np.__version__)             # 1.23.0
print(pd.__version__)             # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/dacon/따릉이/'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)             # [715 rows x 1 columns]

print(train_csv.shape)            #(1459, 10)
print(test_csv.shape)             #(715, 9)
print(submission_csv.shape)       #(715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')           # Nan = 결측치              이상치 ex. 서울의 온도 41도

print(train_csv.info())           # 결측치 확인

print(train_csv.describe())

########################   결측치 처리 1. 삭제   ######################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum())     # 위 함수와 똑같음

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [1328 rows x 10 columns]

########################   결측치 처리 2. 평균값 넣기   ######################
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum()) 
# print(train_csv.info()) 

########################   테스트 데이터의 결측치 확인   ######################
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null

#====================== x 와 y 데이터를 나눠준다 =========================#
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
                #  count라는 axis=1 열 삭제, 참고로 행 삭제는 axis=0
print(x)                                 # [1459 rows x 9 columns]
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다
print(y.shape)                           #(1459,)

"""
# 'Feature' = (['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#         'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#         'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'])

# x = train_csv.drop(columns=['count'])
# y = train_csv['count']
# print(x.shape, y.shape)

# x = x.fillna(x.mean())
"""

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=123
    )                           # 47, 74, 5917


# 2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=9, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=1,
          validation_split=0.2)

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
plt.title('따릉이 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()

'''
# 4. 평가, 예측
print('=====================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])  #정답 도출


def rmse(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = rmse(y_test, results)
r2 = r2_score(y_test, results)

#===========결측치 mean (평균값) 넣은 결과======== 0521_1500 ===========#  r2 0.58 이상 loss 2400.0 이하
# loss: 2410.244140625
# rmse: 49.09423916686265
# r2: 0.5956142502873206

#===========결측치 drop (삭제값) 넣은 결과===================# 0521_1309
# loss: 2197.97265625
# rmse: 46.88254401927481
# r2: 0.653459188934668

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) 
# train 데이터의 shape와 동일한 컬럼을 확인하고 넣는다
                        # x_train.shape:(N, 9) N = Nan
print(y_submit.shape)   # (715, 1)

#============== submission.csv 파일 만들기 // count컬럼값만 넣어주기
print(submission_csv)
submission_csv['count'] = y_submit
print(submission_csv)

print('loss:', loss)
print('rmse:', rmse)
print('r2:', r2)


#======== csv파일 만들기
submission_csv.to_csv(path + 'submission_0522_1008.csv')  # 날짜와 시간을 기입해서 가독성을 높임
'''
# 0521_1518
# loss: 2341.167236328125
# rmse: 48.38560791684631
# r2: 0.6135951594542333

# 0521_1537 relu 사용 후
#loss: 2097.1630859375
# rmse: 45.79479319852236
# r2: 0.6538675162445757

# 0521_1539 relu 사용 후
# loss: 1786.91943359375
# rmse: 42.27196855154464
# r2: 0.7050726150004163

# 0521_1544 relu 사용 후   ###### 123
# loss: 1665.5240478515625
# rmse: 40.81083420709372
# r2: 0.7251086343479001

# 0521_1550 relu 사용 후
# loss: 1756.5977783203125
# rmse: 41.91178570633133
# r2: 0.7100771239298279

# 0521_1608   47
# loss: 1747.656982421875
# rmse: 41.804986194692
# r2: 0.7244577400574386

# 0521_1608   47        #######
# loss: 1595.4884033203125
# rmse: 39.943563183108616
# r2: 0.748449227845091

# 0521_1625       5917
# loss: 1859.9239501953125
# rmse: 43.12683508281393
# r2: 0.7404432318103198

# 0521_1636     
# loss: 1695.0438232421875
# rmse: 41.170910041586076
# r2: 0.7202364772363529

# 0521_1636 123
# loss: 1315.598388671875
# rmse: 36.27117443017366
# r2: 0.8253203573174313

# 0522_0940  123123
# loss: 1754.4737548828125
# rmse: 41.88643721624144
# r2: 0.7670482902225412

# 0522_0945   123123
# loss: 1571.2498779296875
# rmse: 39.638996156573334
# r2: 0.7913759585464243

# 0522_0948  123123
# loss: 1501.933349609375
# rmse: 38.75478328134101
# r2: 0.800579553477959

# 0522_0951 123123
# loss: 1681.129638671875
# rmse: 41.001584411222986
# r2: 0.7767865614237232

# 0522_0958  9741
# loss: 1359.06298828125
# rmse: 36.865473042450844
# r2: 0.8074388389285558

# 0522_1008  123
# loss: 2072.73583984375
# rmse: 45.52730877014475
# r2: 0.706320905990186

########################################################################################
# validation_split 사용 후
# loss: 1743.0731201171875
# rmse: 41.750126777666765
# r2: 0.7123093472400246