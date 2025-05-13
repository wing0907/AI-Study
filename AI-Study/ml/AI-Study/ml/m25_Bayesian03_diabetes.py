import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 준비
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
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y
)

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

path = './_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values(
                        'rank_test_score', # rank_test_score를 기준으로 오름차순 정렬
                        ascending=True).to_csv(path + 'm25_03_diabetes_hgs_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm25_03_diabetes_best_model.joblib')


# 6. 결과 출력
print("\n✅ 최적 하이퍼파라미터:")
for k, v in best_params.items():
    print(f"{k}: {v:.4f}")
print(f"\n📈 Best R² Score (BayesOpt): {optimizer.max['target']:.4f}")
print(f"📉 MSE (최종 평가): {mse:.4f}")
print(f"R² (최종 평가): {r2:.4f}")
print(f"⏱️ Optimization Time: {round(end - start, 2)} seconds")


# ✅ 최적 하이퍼파라미터:
# learning_rate: 0.2403
# max_depth: 3.0383
# min_child_weight: 3.1698
# subsample: 0.5000
# colsample_bytree: 0.7585
# reg_lambda: 4.3605
# reg_alpha: 9.9526

# 📈 Best R² Score (BayesOpt): 0.2462
# 📉 MSE (최종 평가): 0.1826
# R² (최종 평가): 0.2462
# ⏱️ Optimization Time: 6.02 seconds