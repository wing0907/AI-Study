import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
from sklearn.ensemble import HistGradientBoostingRegressor

print(sk.__version__) # 0.24.2      --> 1.6.1

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델구성
# model = RandomForestRegressor()
# allAlgorithms = all_estimators(type_filter='regressor')

# print('allAlgorithms : ', allAlgorithms)
# print('모델의 갯수 : ', len(allAlgorithms)) # 모델의 갯수 :  55
# print(type(allAlgorithms))                 # <class 'list'>


# max_name = ""
# max_score = 0

# for (name, algorithms) in allAlgorithms:
#     try:
#         model = algorithms()
#         model.fit(x_train, y_train)
#         results = model.score(x_test, y_test)
#         print(name, '의 정답률 :', results)

#         if results > max_score:
#             max_score = results
#             max_name = name
#     except:
#         print(name, "은(는) 에러뜬 분!!!")

# print("============================================")
# print("최고모델 :", max_name, max_score)
# print("============================================")

# 최고모델 : HistGradientBoostingRegressor 0.8401018022092235


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

# R² scores: [0.82258725 0.82817146 0.84621976 0.82975847 0.82987473]
# 평균 R² : 0.8313
# 예측값: [4.67953286 1.03033285 2.87018604 0.71903376 1.03638733]
# 실제값: [5.00001 0.707   2.722   1.047   0.938  ]
# cross_val_predict R²: 0.8028
# cross_val_predict MSE: 0.251