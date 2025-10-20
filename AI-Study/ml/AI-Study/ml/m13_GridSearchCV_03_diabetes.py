import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
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

n_split = 5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = KFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태일 때 성능이 더 좋을 때도 있다.


parameters = [
    {"C": [1, 10, 100], "kernel": ['linear'], "degree": [3, 4]},  # poly에는 degree 사용
    {"C": [1, 10], "kernel": ['rbf'], "gamma": [0.01, 0.001]},
    {"C": [1, 10], "kernel": ['poly'], "degree": [2, 3], "gamma": [0.01]}
]

# 2. 모델
model = GridSearchCV(SVR(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1)                                  

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


# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최적의 매개변수 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
# best_score :  0.9916666666666668
# model.score :  0.9
# accuracy_score :  0.9
# time :  1.907 seconds