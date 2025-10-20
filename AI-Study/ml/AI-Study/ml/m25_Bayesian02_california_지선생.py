from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 데이터
from sklearn.datasets import fetch_california_housing
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=333, train_size=0.8)

# 베이지안 범위 설정
bayesian_params = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'n_estimators': (100, 500),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# 목적 함수 정의
def y_function(learning_rate, max_depth, n_estimators, subsample, colsample_bytree):
    model = XGBRegressor(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=333,
        n_jobs=-1
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = r2_score(y_test, pred)
    return score  # or -mse for minimization

# 최적화 시작
optimizer = BayesianOptimization(f=y_function, pbounds=bayesian_params, random_state=333)

start = time.time()
optimizer.maximize(init_points=3, n_iter=10)
end = time.time()

print('최적 파라미터:', optimizer.max['params'])
print('최적 스코어:', optimizer.max['target'])
print('시간:', round(end - start, 2), '초')
