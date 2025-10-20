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
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Split x and y
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

# 3. Replace 0s with NaN (only in specific columns)
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_not_allowed] = x[zero_not_allowed].replace(0, np.nan)

# 4. Fill NaNs with mean
x = x.fillna(x.mean())

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

# acc :  [0.81904762 0.75       0.75961538 0.71153846 0.75961538] 
# 평균 acc :  0.76
# ID
# TRAIN_319    0
# TRAIN_649    0
# TRAIN_041    0
# TRAIN_592    0
# TRAIN_644    1
#             ..
# TRAIN_397    0
# TRAIN_362    1
# TRAIN_146    0
# TRAIN_268    0
# TRAIN_150    0
# Name: Outcome, Length: 131, dtype: int64
# [0 0 1 0 1 0 1 1 0 0 0 1 0 1 1 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0
#  0 0 0 1 1 1 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1 1 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1
#  0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 1 0 0 0 0 0 0 0 0
#  0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0]
# cross_val_predict ACC :  0.6717557251908397