import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']    

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델
model = MLPClassifier()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores ,'\n평균 acc : ', round(np.mean(scores), 4))


y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_pred)


acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

# acc :  [0.78618321 0.77881022 0.78282828 0.77535354 0.7910101 ] 
# 평균 acc :  0.7828
# id
# 58448    Class_9
# 53464    Class_8
# 9337     Class_2
# 56652    Class_8
# 3262     Class_2
#           ...
# 15271    Class_2
# 36275    Class_6
# 3881     Class_2
# 7116     Class_2
# 22087    Class_3
# Name: target, Length: 12376, dtype: object
# ['Class_9' 'Class_9' 'Class_2' ... 'Class_2' 'Class_2' 'Class_2']
# cross_val_predict ACC :  0.7520200387847447