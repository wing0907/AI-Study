import numpy as np
import pandas as pd
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
y = train_csv['count']

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


# XGBRegressor R2 : 0.8116
# RandomForestRegressor R2 : 0.8243
# CatBoostRegressor R2 : 0.8412
# LGBMRegressor R2 : 0.8214

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
print("{0}스태킹 결과 : ".format(model2.__class__.__name__), score2)         


# RandomForestRegressor스태킹 결과 :  0.8085724782419615

