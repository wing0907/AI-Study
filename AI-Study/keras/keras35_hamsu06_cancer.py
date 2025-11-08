import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
# model = Sequential()
# model.add(Dense(64, input_dim=30, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  # 이진분류는 무조건 마지막 activation='sigmoid' 이다. node는 1개

input1 = Input(shape=(30,))
dense1 = Dense(64, activation='relu')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(128, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(16, activation='relu')(dense6)
dense8 = Dense(16, activation='relu')(dense7)
output1 = Dense(1, activation='sigmoid')(dense8)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()