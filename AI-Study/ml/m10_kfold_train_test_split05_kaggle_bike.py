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
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count']    


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


# R² scores: [0.9992969  0.99943476 0.99951377 0.99934772 0.99945237]
# 평균 R² : 0.9994
# 예측값: [186.20585138   4.75058234  89.71875559  62.95838752 151.97403439]
# 실제값: datetime
# 2012-03-07 21:00:00    188
# 2012-04-12 03:00:00      5
# 2011-02-06 10:00:00     89
# 2011-12-05 23:00:00     62
# 2011-08-08 14:00:00    150
# Name: count, dtype: int64
# cross_val_predict R²: 0.9979
# cross_val_predict MSE: 67.8756