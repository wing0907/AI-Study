import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # 반드시 먼저 import
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import joblib


# 1. 데이터
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count']    

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    # stratify=y
)


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {"C": [1, 10, 100], "kernel": ['linear'], "degree": [3, 4]}, 
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

model = HalvingRandomSearchCV(xgb, parameters, cv=kfold,       # 42 * 5 = 210번
                     verbose=1,
                     refit=True, # 다시 훈련                         # 1번
                     n_jobs=1, # -1 = CPU를 FULL로 돌리겠다라는 뜻
                     random_state=333,
                    #  n_iter=11,     # 10 이 default 임
                    # factor=3,           # 배수 : min_resources * factor // default 3임
                    #                     #       n_candidastes / factor
                    # min_resources=30,   # 1 iter때의 최소 훈련 행의 개수
                    # max_resources=120, # 데이터 행의 개수(n_samples)
                    )                                              # 총 271번

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('최적의 매개변수 : ', model.best_estimator_) # 최적의 파라미터가 출력 됨 (지정한 것 중에 제일 좋은 것)
print('최적의 매개변수 : ', model.best_params_) # 최적의 파라미터가 출력 됨 (전체 중에 제일 좋은 것)

# 4. 평가, 예측
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



print(pd.DataFrame(model.cv_results_).sort_values(
                        'rank_test_score', # rank_test_score를 기준으로 오름차순 정렬
                        ascending=True)) # .to_csv(path + 'm15_gs_cv_results.csv')

print(pd.DataFrame(model.cv_results_).columns)
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_learning_rate', 'param_max_depth', 'param_n_estimators',
#        'param_min_child_weight', 'params', 'split0_test_score',
#        'split1_test_score', 'split2_test_score', 'split3_test_score',
#        'split4_test_score', 'mean_test_score', 'std_test_score',
#        'rank_test_score'],
#       dtype='object')


path = './_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values(
                        'rank_test_score', # rank_test_score를 기준으로 오름차순 정렬
                        ascending=True).to_csv(path + 'm22_05_kaggle_bike_hgs_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm22_05_kaggle_bike_best_model.joblib')

# best_score (train CV r² 평균) : 0.9593823432922364
# model.score (test r²) : 0.9991820454597473
# MSE : 25.564
# R² : 0.9992
# time : 18.6862 seconds
