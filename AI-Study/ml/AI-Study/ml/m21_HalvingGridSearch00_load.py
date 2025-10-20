import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import joblib


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)


n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {"n_estimators":[100,500], "max_depth":[6,10,12],
     'learning_rate' :[0.1, 0.01, 0.001]},                             # 18번
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},     # 12번
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]} # 12번
]                                                                   # 총 42번


path = './_save/m15_cv_results/'
model = joblib.load(path + 'm18_06_cancer_best_model.joblib')


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

# best_score :  0.9692307692307693
# model.score :  0.9824561403508771
# accuracy_score :  0.9824561403508771
# time :  24.3587 seconds

# model.score :  0.9824561403508771
# accuracy_score :  0.9824561403508771

print(model)
print(type(model)) 
