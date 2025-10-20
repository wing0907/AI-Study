import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error


#  1. 데이터
x1_datasets = np.array([range(100), range(301,401)]).T                                     # (100, 2)
            # 삼성전자 종가, 하이닉스 종가.
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()    # (100, 3)
            # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401), range(77, 177), range(33, 133)]).T

y1 = np.array(range(2001, 2101))                                                            # (100,)
            # 화성의 화씨 온도.
y2 = np.array(range(13001, 13101))
            # 비트코인 가격
            
x3_datasets = x3_datasets.reshape(100, 4)
# print(x3_datasets.shape)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, test_size=0.3, random_state=190
)

#  2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
output1 = Dense(50, activation='relu', name='ibm5')(dense4) # hidden layer 라서 임의로 정하면 됨. y의 개수(3) 안적어도 됨.
# model1 = Model(inputs=input1, outputs=output1)
# model1.summary()

#  2-2 모델
input2 = Input(shape=(3, ))
dense21 = Dense(100, activation='relu', name='ibm21')(input2)
dense22 = Dense(50, activation='relu', name='ibm22')(dense21)
output2 = Dense(30, activation='relu', name='ibm23')(dense22)
# model2 = Model(inputs=input2, outputs=output2)
# model2.summary()

#  2-3 모델
input3 = Input(shape=(4, ))
dense31 = Dense(50, activation='relu', name='ibm31')(input3) 
dense32 = Dense(80, activation='relu', name='ibm32')(dense31) 
dense33 = Dense(100, activation='relu', name='ibm33')(dense32) 
output3 = Dense(40, activation='relu', name='ibm34')(dense33) 


#  2-4 모델 합치기      # concatenate 대신 Concatenate 쓰기
from keras.layers.merge import concatenate, Concatenate  # 소문자는 함수, 대문자는 클래스.
# 병합
merged = Concatenate()([output1, output2, output3])  # 한 번 더 섞어줌으로서 실질적 모델이 6개가 된다.

shared = Dense(40)(merged)
shared = Dense(20)(shared)

# 출력 2개
output11 = Dense(1, name='y1_output')(shared)       # like this  yo, like this!!
output22 = Dense(1, name='y2_output')(shared)

model = Model(inputs=[input1, input2, input3], outputs=[output11, output22])

model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=100, batch_size=8,
          verbose=1, validation_split=0.1, callbacks=[es],)

# 평가 및 예측 (다중 출력 대응)
results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('Loss:', results)

# 예측 수행
[y1_pred, y2_pred] = model.predict([x1_test, x2_test, x3_test])
y1_pred = y1_pred.reshape(-1,)
y2_pred = y2_pred.reshape(-1,)

print('예측값 샘플 (y1):', y1_pred[:5])
print('예측값 샘플 (y2):', y2_pred[:5])

# R2 score 평가
r2_1 = r2_score(y1_test, y1_pred)
r2_2 = r2_score(y2_test, y2_pred)
print('R2 score (y1):', round(r2_1, 4))
print('R2 score (y2):', round(r2_2, 4))

# 예측
x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T

# 미래 데이터 예측
y1_future, y2_future = model.predict([x1_pred, x2_pred, x3_pred])
y1_future = y1_future.reshape(-1,)
y2_future = y2_future.reshape(-1,)

print('미래 예측값 (y1):', y1_future)
print('미래 예측값 (y2):', y2_future)


# Loss: [0.8116539716720581, 0.44512349367141724, 0.36653050780296326]
# 예측값 샘플 (y1): [2063.3455 2073.5496 2026.135  2047.8423 2041.6393]
# 예측값 샘플 (y2): [13063.278 13073.663 13027.249 13048.267 13042.262]
# R2 score (y1): 0.9994
# R2 score (y2): 0.9995
# 미래 예측값 (y1): [2098.5586 2103.7625 2109.0757 2114.4824 2119.8894 2125.2957]
# 미래 예측값 (y2): [13091.998 13121.25  13151.073 13181.36  13211.649 13241.937]