import numpy as np
from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y
)


n_split = 5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = KFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태일 때 성능이 더 좋을 때도 있다.

parameters = [
    {"C": [1, 10, 100], "kernel": ['linear'], "degree": [3, 4]},  # poly에는 degree 사용
    {"C": [1, 10], "kernel": ['rbf'], "gamma": [0.01, 0.001]},
    {"C": [1, 10], "kernel": ['poly'], "degree": [2, 3], "gamma": [0.01]}
]
                                                   

# 2. 모델
# xgb = XGBRegressor()
xgb = XGBRegressor(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    verbosity=2   # 자세한 로그 출력
)
model = GridSearchCV(xgb, parameters, cv=kfold,       # 42 * 5 = 210번
                     verbose=1,
                     refit=True, # 다시 훈련                         # 1번
                     n_jobs=1, 
                     )                                              # 총 271번

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 : ', model.best_estimator_) # 최적의 파라미터가 출력 됨 (지정한 것 중에 제일 좋은 것)
print('최적의 매개변수 : ', model.best_params_) # 최적의 파라미터가 출력 됨 (전체 중에 제일 좋은 것)
print('best_score (train CV r² 평균) :', model.best_score_) # train에서 가장 좋은 score 가 나옴

# 4. 평가, 예측
score = model.score(x_test, y_test)  # R²
print('model.score (test r²) :', score)

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE :', round(mse, 4))
print('R² :', round(r2, 4))
print('time :', round(end - start, 4), 'seconds')

# 최적의 매개변수 :  XGBRegressor(C=1, base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=0.01, gpu_id=0, grow_policy=None, importance_type=None,
#              interaction_constraints=None, kernel='rbf', learning_rate=None,
#              max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=None, ...)
# 최적의 매개변수 :  {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
# best_score (train CV r² 평균) : 0.8312546852946104
# model.score (test r²) : 0.8291799019140216
# MSE : 0.2176
# R² : 0.8292
# time : 21.3432 seconds