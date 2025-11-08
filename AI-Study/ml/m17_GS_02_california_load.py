import numpy as np
from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import joblib


# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y
)


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {"C": [1, 10, 100], "kernel": ['linear'], "degree": [3, 4]},  # poly에는 degree 사용
    {"C": [1, 10], "kernel": ['rbf'], "gamma": [0.01, 0.001]},
    {"C": [1, 10], "kernel": ['poly'], "degree": [2, 3], "gamma": [0.01]}
]
                                                                     # 총 42번


path = './_save/m15_cv_results/'
model = joblib.load(path + 'm16_california_best_model.joblib')


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('r2 : ', r2_score(y_test, y_pred))


# model.score :  0.8285390039629126
# r2 :  0.8285390039629126

print(model)
# XGBRegressor(C=1, base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, degree=3, device=None,
#              early_stopping_rounds=None, enable_categorical=False,
#              eval_metric=None, feature_types=None, gamma=None, gpu_id=0,
#              grow_policy=None, importance_type=None,
#              interaction_constraints=None, kernel='linear', learning_rate=None,
#              max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              multi_strategy=None, ...)

print(type(model))  # <class 'xgboost.sklearn.XGBRegressor'>
