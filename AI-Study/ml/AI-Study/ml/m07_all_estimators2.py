import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__) # 0.24.2      --> 1.6.1

# x, y = fetch_california_housing(return_X_y=True)
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='classifier')

print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms)) # 모델의 갯수 :  55
print(type(allAlgorithms))                 # <class 'list'>



max_name = ""
max_score = 0

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        print(name, '의 정답률 :', results)

        if results > max_score:
            max_score = results
            max_name = name
    except:
        print(name, "은(는) 에러뜬 분!!!")

print("============================================")
print("최고모델 :", max_name, max_score)
print("============================================")

# 최고모델 : ExtraTreesClassifier 0.975
