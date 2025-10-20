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

                                                        # \ 는 줄바꿈. 파이썬 기초인데 안됨.. 왜 나만 안됨.. 걍 쓰지마셈..ㅋㅋ
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            x1_datasets, x2_datasets, x3_datasets, y1, y2, 
            test_size=0.3, random_state=333
)

#  2-1 모델
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name='ibm1')(input1)
dense2 = Dense(20, activation='relu', name='ibm2')(dense1)
dense3 = Dense(30, activation='relu', name='ibm3')(dense2)
dense4 = Dense(40, activation='relu', name='ibm4')(dense3)
output1 = Dense(50, activation='relu', name='ibm5')(dense4) # hidden layer 라서 임의로 정하면 됨. y의 개수(3) 안적어도 됨.

#  2-2 모델
input2 = Input(shape=(3, ))
dense21 = Dense(10, activation='relu', name='ibm21')(input2)
dense22 = Dense(5, activation='relu', name='ibm22')(dense21)
output2 = Dense(10, activation='relu', name='ibm23')(dense22)

#  2-3 모델
input3 = Input(shape=(4, ))
dense31 = Dense(10, activation='relu', name='ibm31')(input3) 
dense32 = Dense(5, activation='relu', name='ibm32')(dense31) 
output3 = Dense(3, activation='relu', name='ibm34')(dense32) 


#  2-4 모델 합치기
from keras.layers.merge import concatenate, Concatenate  
merge1 = concatenate([output1, output2, output3])  # 한 번 더 섞어줌으로서 실질적 모델이 6개가 된다.

merge2 = Dense(40)(merge1)
merge3 = Dense(20)(merge2)
middle_output = Dense(1, name='last')(merge3)

# model = Model(inputs=[input1, input2], outputs=last_output)
# model.summary()


#  2-5 분리1 -> y1     
#  모델 구성해도 되고 바로 아웃풋 때려도 됨
last_output11 = Dense(10, name='last11')(middle_output)
last_output12 = Dense(10, name='last12')(last_output11)
last_output13 = Dense(1, name='last13')(last_output12)  # last_output13 이 첫번재 y가 됨

#  2-6 분리2 -> y2     
last_output21 = Dense(1, name='last21')(middle_output)

model = Model(inputs=[input1, input2, input3], outputs=[last_output13, last_output21])

#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae') # mae 사용하면 loss가 5개 나옴 4번째가 1번째 mae, 5번이 2번재 mae

#  es, mcp 는 쌤이 귀찮아서 생략하심
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train],
          epochs=100, batch_size=32, verbose=1,
          validation_split=0.1, # 이거하면 자세하게 출력됨. 참고가능
          )

#  4. 평가, 예측
results = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print(results)

# [15417.08984375, 94.26889038085938, 15322.8212890625]   
# loss가 3개가 나온 이유: 2번째 + 3번째 = 1번째 loss. // 1번째 loss = 전체 loss