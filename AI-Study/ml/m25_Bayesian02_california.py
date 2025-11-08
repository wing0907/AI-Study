import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')



# 1. 데이터 준비
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=333, shuffle=True
)

# 2. XGBRegressor에 맞는 파라미터 범위 정의
bayesian_params = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0, 10),
    'reg_alpha': (0, 10),
}

# 3. 목적 함수 정의
def xgb_evaluate(learning_rate, max_depth, min_child_weight,
                 subsample, colsample_bytree, reg_lambda, reg_alpha):
    params = {
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'n_estimators': 100,
        'random_state': 333,
        'n_jobs': -1
    }

    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    return score  # ➤ maximize 대상

# 4. 베이지안 최적화 수행
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds=bayesian_params,
    random_state=333,
    verbose=2
)

start = time.time()
optimizer.maximize(init_points=5, n_iter=20)
end = time.time()

# 5. 최적 파라미터로 재학습 및 평가
best_params = optimizer.max['params']
model = XGBRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_lambda=best_params['reg_lambda'],
    reg_alpha=best_params['reg_alpha'],
    n_estimators=100,
    random_state=333,
    n_jobs=-1
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 6. 결과 출력
print("\n✅ 최적 하이퍼파라미터:")
for k, v in best_params.items():
    print(f"{k}: {v:.4f}")
print(f"\n📈 Best R² Score (BayesOpt): {optimizer.max['target']:.4f}")
print(f"📉 MSE (최종 평가): {mse:.4f}")
print(f"R² (최종 평가): {r2:.4f}")
print(f"⏱️ Optimization Time: {round(end - start, 2)} seconds")


# ✅ 최적 하이퍼파라미터:
# learning_rate: 0.1177
# max_depth: 8.8015
# min_child_weight: 2.0353
# subsample: 0.6114
# colsample_bytree: 0.7509
# reg_lambda: 4.0356
# reg_alpha: 1.1462

# 📈 Best R² Score (BayesOpt): 0.8386
# 📉 MSE (최종 평가): 0.2057
# R² (최종 평가): 0.8386
# ⏱️ Optimization Time: 10.91 seconds