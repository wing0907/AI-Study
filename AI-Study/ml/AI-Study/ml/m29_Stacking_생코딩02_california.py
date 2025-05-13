import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=331,
    # stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2-1. 모델
xgb = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)      # default epochs=1000
lg = LGBMRegressor(verbose=0)

models = [xgb, rf, cat, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train) # 값을 가지고 다시 훈련시켜야 하기 때문에 x_train 으로 함.
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append((y_test_pred))

    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__       # 언더바 2개 모델의 클래스 이름을 볼 수 있음
    print('{0} R2 : {1:.4F}'.format(class_name, score))


# XGBRegressor R2 : 0.8334
# RandomForestRegressor R2 : 0.8016
# CatBoostRegressor R2 : 0.8495
# LGBMRegressor R2 : 0.8301

x_train_new = np.array(train_list).T
# print(x_train_new)
print(x_train_new.shape)  # (16512, 4)

x_test_new = np.array(test_list).T
print(x_test_new.shape)   # (4128, 4)


# 2-2. 모델
model2 = RandomForestRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print("스태킹 결과 : ", score2)         # 스태킹 결과 :  0.78316033476479 현재 과적합 상태. 파라미터 튜닝이 안되어있음



