import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from xgboost.callback import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import random
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

seed = 333
random.seed(seed)
np.random.seed(seed)


# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.8,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
# model = DecisionTreeRegressor()
model = BaggingRegressor(DecisionTreeRegressor(),       #wrapping 함
                         n_estimators=100,
                         n_jobs=-1,
                         random_state=seed,
                        #  bootstrap=False, 데이터를 중복해서 사용할 것인지 결정. 사용하는게 디폴트
                         )
# model = RandomForestRegressor(random_state=seed,)



# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# DecisionTreeRegressor
# 최종점수 :  0.5934739286037887


# BaggingRegressor  // bootstrap=True  True가 Default임 // 샘플데이터 중복 허용
# 최종점수 :  0.799631827482626

# BaggingRegressor  // bootstrap=False // 샘플데이터 중복 비허용
# 최종점수 :  0.6228984968458924


# RandomForestRegressor
# 최종점수 :  0.7990469782823068
