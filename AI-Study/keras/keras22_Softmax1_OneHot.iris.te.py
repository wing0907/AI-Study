import numpy as np
import pandas as pd
from sklearn.datasets import load_iris          # 다중분류
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time
# 1. 데이터
datasets = load_iris()



x = datasets.data
y = datasets.target
print(x.shape, y.shape)                         # (150, 4) (150,)

print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
print(np.unique(y, return_counts=True))         # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0    50
# 1    50
# 2    50
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50


#####################################################################
'''
1. sklearn - OneHotEncoder              재익은 재현의 두배가 아니야.   # 다중분류에서는 "반드시" OneHotEncoding을 사용하고 Y 만 적용한다!!!!!!!!

from sklearn.preprocessing import OneHotEncoder
import numpy as np

y = np.array([0, 1, 2, 1, 0])

encoder = OneHotEncoder(sparse=False)  # sparse=False면 배열 반환
y_reshaped = y.reshape(-1, 1)          # 2D 형태로 만들어야 함
y_onehot = encoder.fit_transform(y_reshaped)

print(y_onehot)

장점: 데이터가 2D 배열이어야 하고, 범주형 데이터를 원-핫 벡터로 변환할 때 많이 씀.
sparse=False가 없으면 희소 행렬 반환.
'''
# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1, 1)                # reshape 바뀌지 않아야 할것 2가지 =  1. 안에 들어간 값  / 2. 순서

# ohe = OneHotEncoder()               # metrics형태를 받기때문에 (N, 1)로 reshape하고 해야한다.
# y = ohe.fit_transform(y)#.toarray()  # .toarray()를 쓰면 numpy 형태로 바뀜
# print(y)                            # 희소행렬방식
# print(y.shape)   # (150, 3)
# print(type(y))                      # <class 'scipy.sparse.csr.csr_matrix'>
# y = y.toarray()                     # scipy를 numpy로 변환


# ohe = OneHotEncoder(sparse=False)   # numpy형태 출력, 디폴트는 True
# y = ohe.fit_transform(y)
# print(y)

#####################################################################
'''
2. pandas - get_dummies

import pandas as pd

y = pd.Series([0, 1, 2, 1, 0])
y_onehot = pd.get_dummies(y)

print(y_onehot)

장점: 데이터가 Series 형태일 때 바로 사용 가능.
결과는 DataFrame 형태.
쉽게 원-핫 인코딩이 가능하고, 다시 NumPy 배열로 바꿀 수도 있음: y_onehot.values
'''
y = pd.get_dummies(y)
print(y)
print(y.shape)      # (150, 3)
#####################################################################
'''
3. Keras - to_categorical                           0이 아닌 1부터 시작하는 데이터는 0이라는 컬럼이 새로 생겨버린다.
                                                    그래서 항상 찍어보고 .shape 확인해보기
from tensorflow.keras.utils import to_categorical
import numpy as np

y = np.array([0, 1, 2, 1, 0])
y_onehot = to_categorical(y)

print(y_onehot)

장점: 정수 레이블 배열을 바로 받아서 원-핫 인코딩.
TensorFlow/Keras 모델 학습 시 가장 간단하고 직관적.
'''
#####################################################################
"""
<비교 요약>
방법	            입력 타입	                     출력 타입	        특징
OneHotEncoder	    2D 배열 (n_samples, 1)	         NumPy 배열	      sklearn 스타일, 복잡하지만 유연함
pd.get_dummies	    Pandas Series 또는 DataFrame	 DataFrame	      간단, 판다스 친화적
to_categorical	    1D 정수 배열	                 NumPy 배열	      케라스 친화적, 딥러닝에 적합
"""
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150, 3)
#####################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=333,
)


#  2. 모델구성
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, input_dim=4, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

#  3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=100,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train, #validation_data=(x_test, y_test),
        epochs=1000, batch_size=8, verbose=1, validation_split=0.1,
        callbacks=[es])
end_time = time.time()

#  4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss[0])
print('acc:', loss[1])

################ accuracy_score를 사용해서 출력해볼것!!!1 ###############
from sklearn.metrics import accuracy_score
# 힌트는 argmax
y_pred = model.predict(x_test)
y_test = y_test.to_numpy()  # pandas 형태이기 때문에  numpy 로 변환해준다.
print(y_test)
print(y_pred)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred)
print('acc:' , round(acc, 4))

# loss: 0.03978443890810013
# acc: 1.0

# loss: 0.18620748817920685
# acc: 0.9333