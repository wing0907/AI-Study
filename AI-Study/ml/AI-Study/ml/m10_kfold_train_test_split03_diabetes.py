import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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
    x, y, shuffle=True, random_state=123, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingRegressor()

# 3. 훈련 및 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('R² scores:', scores)
print('평균 R² :', round(np.mean(scores), 4))

# 4. 예측
y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
print('예측값:', y_pred[:5])
print('실제값:', y_test[:5])

# 5. 최종 평가
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('cross_val_predict R²:', round(r2, 4))
print('cross_val_predict MSE:', round(mse, 4))


# R² scores: [0.16716669 0.25598835 0.16725111 0.13806053 0.28305446]
# 평균 R² : 0.2023
# 예측값: [0.40831443 0.05471067 0.38875979 0.47284437 0.36098559]
# 실제값: ID
# TRAIN_379    1
# TRAIN_226    0
# TRAIN_043    0
# TRAIN_476    1
# TRAIN_426    0
# Name: Outcome, dtype: int64
# cross_val_predict R²: 0.0138
# cross_val_predict MSE: 0.2376