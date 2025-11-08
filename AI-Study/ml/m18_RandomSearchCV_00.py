import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import joblib


# 1. 데이터
x, y = load_iris(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)


n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {"n_estimators":[100,500], "max_depth":[6,10,12],
     'learning_rate' :[0.1, 0.01, 0.001]},                             # 18번
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1, 0.01, 0.001]},     # 12번
    {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]} # 12번
]                                                                   # 총 42번

# 2. 모델
xgb = XGBClassifier()
# xgb = XGBClassifier(
#     tree_method='gpu_hist',
#     predictor='gpu_predictor',
#     gpu_id=0,
#     verbosity=2   # 자세한 로그 출력
# )
model = RandomizedSearchCV(xgb, parameters, cv=kfold,       # 42 * 5 = 210번
                     verbose=1,
                     refit=True, # 다시 훈련                         # 1번
                     n_jobs=-1, # CPU를 FULL로 돌리겠다라는 뜻
                     random_state=333,
                     n_iter=11,     # 10 이 default 임
                     )                                              # 총 271번

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('최적의 매개변수 : ', model.best_estimator_) # 최적의 파라미터가 출력 됨 (지정한 것 중에 제일 좋은 것)
print('최적의 매개변수 : ', model.best_params_) # 최적의 파라미터가 출력 됨 (전체 중에 제일 좋은 것)

exit()
# 4. 평가, 예측
print('best_score : ', model.best_score_)    # train에서 가장 좋은 score 가 나옴

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))
print('time : ', round(end - start, 4), 'seconds')

# 최적의 매개변수 :  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=6, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=500, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 최적의 매개변수 :  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
# best_score :  0.95
# model.score :  0.9
# accuracy_score :  0.9
# time :  4.2102 seconds

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
                        ascending=True).to_csv(path + 'm18_00_rs_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm18_00_best_model.joblib')

