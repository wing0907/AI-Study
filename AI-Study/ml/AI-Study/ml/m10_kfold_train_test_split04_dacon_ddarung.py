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
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isna().sum())     # 위 함수와 똑같음
train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다


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


# R² scores: [0.80012484 0.77615919 0.71413807 0.70973021 0.80337364]
# 평균 R² : 0.7607
# 예측값: [ 55.17676847 125.31613473  47.62861752 261.41347087 244.92080122]
# 실제값: id
# 1130     24.0
# 579     113.0
# 517      55.0
# 57      214.0
# 905     218.0
# Name: count, dtype: float64
# cross_val_predict R²: 0.6663
# cross_val_predict MSE: 2438.7991