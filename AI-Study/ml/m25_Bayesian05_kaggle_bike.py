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
import warnings
warnings.filterwarnings('ignore')
import random
import joblib


seed = 333
random.seed(seed)
np.random.seed(seed)


# 1. 데이터
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count']    
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.8,
)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델

bayesian_params = {
    'n_estimators' : (100, 500), # 추가 epoch
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3,10),
    # 'num_leaves' : (24,40),
    # 'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'gamma' : (0, 5),        # 추가
    'subsample' : (0.5, 1),            # 0~1사이인데 2를 주면 'subsample' : max(min(subsample, 1), 0) 하면 0~1사이로 바운더리가 잡힘.
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),    # 추가
    # 'max_bin' : (9, 500),
    'reg_lambda' : (0, 100),        # 무조건 0이상이여야 함 // default 1 // L2 정규화 // 릿지
    'reg_alpha' : (0, 10),          # default 0 // L1 정규화 // 라쏘
}

def xgb_hamsu(n_estimators, learning_rate, max_depth, min_child_weight,
              gamma, subsample, colsample_bytree,
              colsample_bylevel, reg_lambda, reg_alpha):
    params = {
        # 'n_estimators' : 100, # default = 100
        'n_estimators' : int(round(n_estimators)),
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)), # 확인사살. 정수로 바꾸기
        'min_child_weight' : int(round(min_child_weight)),
        'gamma' : gamma, 
        'subsample' : subsample,       # 한계점이 0에서 1사이로 줘야함.
        'colsample_bytree' : colsample_bytree,
        'colsample_bylevel' : colsample_bylevel,
        'reg_lambda' : max(reg_lambda, 0), #이래 하면 relu 처럼 0이상 나옴
        'reg_alpha' : reg_alpha,
        
    }
    
    model = XGBRegressor(**params, n_jobs=-1,  )
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
              verbose = False,
              eval_metric='rmse',
              early_stopping_rounds=30,       
              )
    
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    result = -rmse  # ❗최적화 대상이므로 음수로 반환

    return result

    
    
# 최적화 시작
optimizer = BayesianOptimization(
    f = xgb_hamsu, 
    pbounds=bayesian_params,
    random_state=seed,
    )

n_iter = 200
start = time.time()
optimizer.maximize(init_points=10,
                   n_iter=n_iter)
end = time.time()

print(optimizer.max)
print(n_iter, '번 걸린 시간 : ', round(end - start, 2), '초')

# 200 번 걸린 시간 :  114.41 초
# {'target': 0.8413026501340473,

print('최적 파라미터:', optimizer.max['params'])
print('최적 스코어:', optimizer.max['target'])
print('최적 RMSE:', -optimizer.max['target']) 


path = './_save/m15_cv_results/'
pd.DataFrame(optimizer.cv_results_).sort_values(
                        'rank_test_score', # rank_test_score를 기준으로 오름차순 정렬
                        ascending=True).to_csv(path + 'm25_05_kaggle_bike_hgs_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(optimizer.best_estimator_, path + 'm25_05_kaggle_bike_best_model.joblib')


# 최적 파라미터: {'learning_rate': 0.1, 'max_depth': 10.0, 'min_child_weight': 31.010552411917406, 'subsample': 1.0, 'colsample_bytree': 0.834152991728681, 'reg_lambda': 2.315654246904706, 'reg_alpha': 5.426628500751654}
# 최적 스코어: 0.8413026501340473


# {'target': 0.8480250964468037,
# 200 번 걸린 시간 :  219.24 초
# 최적 파라미터: {'n_estimators': 342.593417772331, 'learning_rate': 0.1, 'max_depth': 10.0, 'min_child_weight': 15.467639989609038, 'gamma': 0.0, 'subsample': 1.0, 'colsample_bytree': 0.5, 'colsample_bylevel': 1.0, 'reg_lambda': 39.770816478876945, 'reg_alpha': 0.0}
# 최적 스코어: 0.8480250964468037