import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import random
import joblib
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestClassifier, RandomForestRegressor, VotingClassifier

seed = 333
random.seed(seed)
np.random.seed(seed)


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.8, stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('XGB' , xgb), ('LG', lg), ('CAT', cat)],
    voting='hard',   # Default
    # voting='soft',
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# soft
# 최종점수 :  0.9824561403508771

# hard
# 최종점수 :  0.9824561403508771