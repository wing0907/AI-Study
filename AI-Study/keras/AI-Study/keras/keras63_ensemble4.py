import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error

# 걍 분기만 함. 이런 모델은 실제로 잘안씀.

#  1. 데이터
x_datasets = np.array([range(100), range(301,401)]).T                                     # (100, 2)
            # 삼성전자 종가, 하이닉스 종가.

y1 = np.array(range(2001, 2101))                                                            # (100,)
            # 화성의 화씨 온도.
y2 = np.array(range(13001, 13101))
            # 비트코인 가격

                                                        # \ 는 줄바꿈. 파이썬 기초
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            x_datasets, y1, y2, 
            test_size=0.3, random_state=333
)

#  2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
output1 = Dense(50, activation='relu', name='ibm5')(dense4) # hidden layer 라서 임의로 정하면 됨. y의 개수(3) 안적어도 됨.


#  2-4 모델 합치기
from keras.layers.merge import concatenate, Concatenate  
merge1 = concatenate([output1])  # 한 번 더 섞어줌으로서 실질적 모델이 6개가 된다.

merge2 = Dense(40)(merge1)
merge3 = Dense(20)(merge2)
middle_output = Dense(1, name='last')(merge3)

#  2-5 분리1 -> y1     
#  모델 구성해도 되고 바로 아웃풋 때려도 됨
last_output11 = Dense(10, name='last11')(middle_output)
last_output12 = Dense(10, name='last12')(last_output11)
last_output13 = Dense(1, name='last13')(last_output12)  # last_output13 이 첫번재 y가 됨

#  2-6 분리2 -> y2     
last_output21 = Dense(1, name='last21')(middle_output)

model = Model(inputs=[input1], outputs=[last_output13, last_output21])

#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae') # mae 사용하면 loss가 5개 나옴 4번째가 1번째 mae, 5번이 2번재 mae

#  es, mcp 는 쌤이 귀찮아서 생략하심
model.fit(x1_train, [y1_train, y2_train],
          epochs=100, batch_size=32, verbose=1,
          validation_split=0.1, # 이거하면 자세하게 출력됨. 참고가능
          )

#  4. 평가, 예측
results = model.evaluate(x1_test, [y1_test, y2_test])
print('Loss:', results)
[y1_pred, y2_pred] = model.predict(x1_test)
r2_1 = r2_score(y1_test, y1_pred)
r2_2 = r2_score(y2_test, y2_pred)
print('R2 score (y1):', round(r2_1, 4))
print('R2 score (y2):', round(r2_2, 4))
print('y1:', y1_pred.flatten())
print('y2:', y2_pred.flatten())
# [337074.53125, 4990.06982421875, 332084.46875, 59.16794204711914, 480.4210205078125]

# Loss: [267557.0, 3626.823974609375, 263930.1875, 50.16120147705078, 428.5537414550781]
# R2 score (y1): -5.1453
# R2 score (y2): -446.2012
# y1: [2053.856  2085.153  2182.5264 2092.1082 2109.496  1932.1399 2026.0352
#  2112.9739 2203.3926 2005.1694 2029.5132 2147.7493 1998.2139 2199.9143
#  1935.6184 2022.5588 2067.7673 1921.7075 2175.5713 1977.3499 2116.4512
#  2064.2893 1925.1848 2074.7222 2133.8418 2078.1992 2095.5867 1893.8878
#  2015.602  1987.7817]
# y2: [13066.745 13265.869 13885.363 13310.118 13420.743 12292.378 12889.747
#  13442.867 14018.112 12756.999 12911.871 13664.115 12712.748 13995.986
#  12314.502 12867.623 13155.245 12226.002 13841.112 12579.999 13464.991
#  13133.121 12248.128 13199.494 13575.616 13221.618 13332.243 12049.005
#  12823.372 12646.375]