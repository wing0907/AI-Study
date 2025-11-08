import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

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


# R² scores: [0.84324009 0.8371523  0.82813665 0.82975886 0.82717565]
# 평균 R² : 0.8331
# 예측값: [2.10938304 0.91913903 1.52384928 1.72399683 4.50578316]
# 실제값: [1.516 0.992 1.345 2.317 4.629]
# cross_val_predict R²: 0.8061
# cross_val_predict MSE: 0.2579