import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 데이터 로드 및 전처리
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 베이지안 탐색 파라미터 공간 정의
bayesian_params = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 12),
    'num_leaves': (20, 150),
    'min_child_samples': (10, 100),
    'min_child_weight': (1e-3, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0, 10),
    'reg_alpha': (0, 10),
}

# 목적 함수 정의
def lgb_hamsu(learning_rate, max_depth, num_leaves,
              min_child_samples, min_child_weight,
              subsample, colsample_bytree,
              reg_lambda, reg_alpha):
    
    params = {
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples': int(round(min_child_samples)),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
    }

    model = lgb.LGBMRegressor(**params, n_estimators=1000)
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        # early_stopping_rounds=50,
        # verbose=0     
    )

    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return -rmse  # 최소화 문제를 최대화로 바꾸기 위해 부호 반전

# Bayesian Optimization 실행
optimizer = BayesianOptimization(
    f=lgb_hamsu,
    pbounds=bayesian_params,
    random_state=42,
    verbose=2
)

start = time.time()
optimizer.maximize(init_points=10, n_iter=50)
end = time.time()

# 결과 출력
print()
print(f"🔍 최적 파라미터: {optimizer.max['params']}")
print(f"✅ 최적 스코어(RMSE 음수): {optimizer.max['target']}")
print(f"✅ 최적 RMSE: {-optimizer.max['target']}")
print(f"⏱️ 소요 시간: {round(end - start, 2)}초")


# 🔍 최적 파라미터: {'learning_rate': 0.033198583802866694, 'max_depth': 11.420623014917382, 'num_leaves': 91.36755636896774, 'min_child_samples': 27.982228141704677, 'min_child_weight': 5.549167146567621, 'subsample': 0.955436828140207, 'colsample_bytree': 0.80733187838962, 'reg_lambda': 5.022705891789624, 'reg_alpha': 3.5357401386282628}
# ✅ 최적 스코어(RMSE 음수): -0.43121707204399984
# ✅ 최적 RMSE: 0.43121707204399984
# ⏱️ 소요 시간: 57.42초