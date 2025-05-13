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
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train_csv['target'])    


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
    # weights=[2,1,1],    # soft voting에서만 돌아감
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# soft
# 최종점수 :  0.815530058177117

# hard
# 최종점수 :  0.8679674008543642