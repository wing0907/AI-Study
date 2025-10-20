import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
# x, y = fetch_covtype(return_X_y=True)
datasets = fetch_covtype()

x = datasets.data
y = datasets['target'] -1

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=331,
    stratify=y,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2-1. 모델
xgb = XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)      # default epochs=1000
lg = LGBMClassifier(verbose=0, verbosity=-1)

models = [xgb, rf, cat, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train) # 값을 가지고 다시 훈련시켜야 하기 때문에 x_train 으로 함.
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append((y_test_pred))

    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__       # 언더바 2개 모델의 클래스 이름을 볼 수 있음
    print('{0} ACC : {1:.4F}'.format(class_name, score))


# XGBClassifier ACC : 0.8722
# RandomForestClassifier ACC : 0.9559
# CatBoostClassifier ACC : 0.8864
# LGBMClassifier ACC : 0.8502

x_train_new = np.column_stack(train_list)
x_test_new = np.column_stack(test_list)



# 2-2. 모델
model2 = RandomForestClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print("{0}스태킹 결과 : ".format(model2.__class__.__name__), score2)     # 이름도 예쁘게 들어감

# RandomForestClassifier스태킹 결과 :  0.9558703303701281

