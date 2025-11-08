# XGBoost
# california
# diabetes

# LGBM
# cancer
# digits

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

RANDOM_STATE = 333
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------
# 1) 데이터 로더
# ---------------------------
def load_dataset(name: str):
    name = name.lower()
    if name == "california":
        d = fetch_california_housing()
        X, y = d.data, d.target
    elif name == "diabetes":
        d = load_diabetes()
        X, y = d.data, d.target
    else:
        raise ValueError("Choose one of: 'california', 'diabetes'")
    return X, y

DATASETS = ["california", "diabetes"]

# ---------------------------
# 2) 파이프라인 (scaler -> pca -> xgb)
# ---------------------------
pipe = make_pipeline([
    ("scaler", "passthrough"),          # 그리드에서 선택
    ("pca", "passthrough"),             # 그리드에서 선택
    ("xgbregressor", XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",             # CPU 빠르게
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="rmse"
    )),
])

# ---------------------------
# 3) 그리드(적당한 규모)
#    - scaler: 꺼짐/MinMax/Standard/Robust
#    - pca: 꺼짐 vs 0.95 분산보존
#    - xgb 하이퍼파라미터
# ---------------------------
param_grid = {
    "scaler": ["passthrough", MinMaxScaler(), StandardScaler(), RobustScaler()],
    "pca": ["passthrough", PCA(n_components=0.95, random_state=RANDOM_STATE)],
    "xgbregressor__n_estimators": [400, 800],
    "xgbregressor__max_depth": [3, 6, 8],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
    # 필요시 정규화도 추가 가능:
    "xgbregressor__reg_lambda": [1.0, 5.0, 10.0],
}

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# GridSearchCV의 scoring은 RMSE 기준으로 보고 싶지만
# 오래된 sklearn 호환 위해 neg_mean_squared_error 사용
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",  # CV에서 MSE 최소화 == RMSE 최소화
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,                        # 베스트로 재학습
)

# ---------------------------
# 4) 실행 & 리포트
# ---------------------------
summary_rows = []

for name in DATASETS:
    print(f"\n========== DATASET: {name} ==========")
    X, y = load_dataset(name)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    grid.fit(x_train, y_train)

    # CV의 best score (neg MSE) → RMSE로 변환
    best_cv_rmse = np.sqrt(-grid.best_score_)
    print(f"[{name}] Best CV RMSE: {best_cv_rmse:.5f}")
    print(f"[{name}] Best Params : {grid.best_params_}")

    # 테스트셋 성능
    y_pred = grid.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[{name}] Test R2  : {r2:.5f}")
    print(f"[{name}] Test RMSE: {rmse:.5f}")
    print(f"[{name}] Test MAE : {mae:.5f}")

    summary_rows.append({
        "dataset": name,
        "best_params": grid.best_params_,
        "cv_rmse": round(best_cv_rmse, 5),
        "test_r2": round(r2, 5),
        "test_rmse": round(rmse, 5),
        "test_mae": round(mae, 5),
    })

print("\n========= SUMMARY =========")
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

'''
========== DATASET: california ==========
Fitting 5 folds for each of 256 candidates, totalling 1280 fits
[california] Best CV RMSE: 0.44737
[california] Best Params : {'pca': 'passthrough', 'scaler': 'passthrough', 'xgb__colsample_bytree': 0.8, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 6, 'xgb__n_estimators': 400, 'xgb__subsample': 0.8}
[california] Test R2  : 0.84472
[california] Test RMSE: 0.44479
[california] Test MAE : 0.29006

========== DATASET: diabetes ==========
Fitting 5 folds for each of 256 candidates, totalling 1280 fits
[diabetes] Best CV RMSE: 57.03577
[diabetes] Best Params : {'pca': PCA(n_components=0.95, random_state=333), 'scaler': StandardScaler(), 'xgb__colsample_bytree': 0.8, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__n_estimators': 200, 'xgb__subsample': 0.8}
[diabetes] Test R2  : 0.38564
[diabetes] Test RMSE: 57.07235
[diabetes] Test MAE : 46.47037

========= SUMMARY =========
   dataset                                                                                                                                                                                                   best_params  cv_rmse  test_r2  test_rmse  test_mae
california                                {'pca': 'passthrough', 'scaler': 'passthrough', 'xgb__colsample_bytree': 0.8, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 6, 'xgb__n_estimators': 400, 'xgb__subsample': 0.8}  0.44737  0.84472    0.44479   0.29006
  diabetes {'pca': PCA(n_components=0.95, random_state=333), 'scaler': StandardScaler(), 'xgb__colsample_bytree': 0.8, 'xgb__learning_rate': 0.05, 'xgb__max_depth': 3, 'xgb__n_estimators': 200, 'xgb__subsample': 0.8} 57.03577  0.38564   57.07235  46.47037
'''