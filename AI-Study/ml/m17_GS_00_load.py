import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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


path = './_save/m15_cv_results/'
model = joblib.load(path + 'm16_best_model.joblib')


# 4. 평가, 예측

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

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

# model.score :  0.9
# accuracy_score :  0.9


print(model)
print(type(model)) 