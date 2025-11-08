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

y = np.array(range(2001, 2101))                                                            # (100,)
            # 화성의 화씨 온도.

x3_datasets = x3_datasets.reshape(100, 4)
# print(x3_datasets.shape)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y, test_size=0.3, random_state=190
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

#  2-2 모델
input3 = Input(shape=(4, ))
dense31 = Dense(50, activation='relu', name='ibm31')(input3) 
dense32 = Dense(80, activation='relu', name='ibm32')(dense31) 
dense33 = Dense(100, activation='relu', name='ibm33')(dense32) 
output3 = Dense(40, activation='relu', name='ibm34')(dense33) 


#  2-4 모델 합치기      # concatenate 대신 Concatenate 쓰기
from keras.layers.merge import concatenate, Concatenate  # 소문자는 함수, 대문자는 클래스.
merge1 = Concatenate(name='mg1')([output1, output2, output3])

merge2 = Dense(40, name='mg2')(merge1)
merge3 = Dense(20, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2, input3], outputs=last_output)
model.summary()

# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_1 (InputLayer)           [(None, 2)]          0           []

#  ibm1 (Dense)                   (None, 10)           30          ['input_1[0][0]']

#  ibm2 (Dense)                   (None, 20)           220         ['ibm1[0][0]']

#  input_2 (InputLayer)           [(None, 3)]          0           []

#  ibm3 (Dense)                   (None, 30)           630         ['ibm2[0][0]']

#  ibm21 (Dense)                  (None, 100)          400         ['input_2[0][0]']

#  ibm4 (Dense)                   (None, 40)           1240        ['ibm3[0][0]']

#  ibm22 (Dense)                  (None, 50)           5050        ['ibm21[0][0]']

#  ibm5 (Dense)                   (None, 50)           2050        ['ibm4[0][0]']

#  ibm23 (Dense)                  (None, 30)           1530        ['ibm22[0][0]']

#  mg1 (Concatenate)              (None, 80)           0           ['ibm5[0][0]',
#                                                                   'ibm23[0][0]']

#  mg2 (Dense)                    (None, 40)           3240        ['mg1[0][0]']

#  mg3 (Dense)                    (None, 20)           820         ['mg2[0][0]']

#  last (Dense)                   (None, 1)            21          ['mg3[0][0]']

# ==================================================================================================
# Total params: 15,231
# Trainable params: 15,231
# Non-trainable params: 0
# __________________________________________________________________________________________________

# dataset이 다를 경우 앙상블을 사용하되 임의로 데이터를 분리해서 사용해도 무방. 성능이 좋을지 안좋을지는 모름.
# 엄밀히 얘기하면 단일 모델임


#  3. 컴파일, 훈련          # 2개 이상은 리스트!
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

model.fit([x1_train, x2_train, x3_train], y_train, epochs=100, batch_size=8,
          verbose=1, validation_split=0.1, callbacks=[es],)


#  4. 평가, 예측
x1_pred = np.array([range(100, 106), range(400, 406)]).T
x2_pred = np.array([range(200, 206), range(510, 516), range(249, 255)]).T
x3_pred = np.array([range(100, 106), range(400, 406), range(177, 183), range(133, 139)]).T




results = model.evaluate([x1_test, x2_test, x3_test], y_test)

y_pred = model.predict([x1_test, x2_test, x3_test])
print('예측값 샘플:', y_pred[:5].flatten())

r2 = r2_score(y_test, y_pred)
print('R2 score:', round(r2, 4))

y_future = model.predict([x1_pred, x2_pred, x3_pred])
print('미래 예측값:', y_future.flatten())  # .flatten() 보기 쉽게 1차원으로 펴는 것

# 예측값 샘플: [2063.1294 2071.2651 2026.5265 2048.626  2042.3118]
# R2 score: 0.9974
# 미래 예측값: [2097.7798 2099.814  2102.335  2104.8555 2107.4634 2110.7566]