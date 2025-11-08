import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer

# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)         #   :Number of Instances: 569  /   :Number of Attributes: 30 numeric, predictive attributes and the class
print(datasets.feature_names)
'''    
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
'''
print(type(datasets))         # <class 'sklearn.utils.Bunch'>

x = datasets.data
y = datasets.target
print(x.shape, y.shape)       # (569, 30) (569,)
print(type(x))                # <class 'numpy.ndarray'>
print(x)
'''
[[1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
 [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
 [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
 ...
 [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
 [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
 [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02]]
'''
print(y)                      # 지도학습 Supervised Learning (답안지가 있는 놈)에는 회귀와 분류가 있다.
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 0 0 0 0 0 1]
'''

# 0과 1의 갯수가 몇개인지 찾아보기. pandas와 numpy에서 쓰는 코드

'''
import pandas as pd

# y를 Series로 변환한 뒤 value_counts()                         방법	    코드	                           출력 형태
y_counts = pd.Series(y).value_counts()                         Pandas	  pd.Series(y).value_counts()	     Series (인덱스: 값, 값: 개수)
print(y_counts)                                                Numpy	  np.unique(y, return_counts=True)	 Dictionary 또는 두 배열

# 1    357
# 0    212
'''

"""
import numpy as np

unique, counts = np.unique(y, return_counts=True)
y_counts_np = dict(zip(unique, counts))
print(y_counts_np)

# {0: 212, 1: 357}
"""
#############################################################

print(np.unique(y, return_counts=True))     # 1. 넘파이로 찾았을 때
# (array([0, 1]), array([212, 357], dtype=int64))

print(pd.value_counts(y))                   # 2. 판다스로 찾았을 때
# 1    357
# 0    212

print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=223,
    shuffle=True, 
)

print(x_train.shape, x_test.shape)      # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)      # (398,) (171,)

# 2. 모델구성.
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진분류는 무조건 마지막 activation='sigmoid' 이다. node는 1개

# 3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam',    # 이진분류는 loss = binary_crossentropy. 
              metrics=['acc'])          # acc = accuracy *tensorflow에서는 metrics에서 제공해준다. accuracy 라고 써도 된다. 표기는 풀네임으로 됨. 훈련에 영향은 안미침.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss', 
    mode = 'min',                    # 최대값 max, 알아서 찾아줘: auto
    patience=20,                     # 이만큼 참을 거다. local minimum(국소 최소값), global minimum(전역 최소값)
    restore_best_weights=True,       #EarlyStopping 의 Default는 False. True = 최적의 weight(가중치) 때 멈춘다. 최소지점 save.
)


import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:44.718308
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")     # string 문자열
print(date)     # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras28_mcp/06_cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_', date, '_', filename])



mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=50, batch_size=8,
          verbose=1,
          validation_split=0.2,
          callbacks=[es, mcp],
          )    
end_time = time.time()
# es 적용한 것과 안한 것. 몇 epoch에서 끊겼는지 기록. restore_best_weights=True 값과 False 값

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)      # [0.11011096835136414, 0.9590643048286438] loss와 accuracy
# acc > 0.97 
# [0.17258593440055847, 0.9649122953414917]
# [0.09645834565162659, 0.9736841917037964]
print("loss : ", results[0])            # loss :  0.10789068043231964
print("acc : ", round(results[1], 4))             # acc :  0.9649122953414917, round 4 = 소수 5번째에서 반올림 acc :  0.9708


y_predict = model.predict(x_test)           # 요래하면 에러 해결됨
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict[:10])

from sklearn.metrics import accuracy_score              
accuracy_score = accuracy_score(y_test, y_predict)
print("acc_score : ", accuracy_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")   



# acc_score :  0.935672514619883
# 걸린시간 :  3.11 초