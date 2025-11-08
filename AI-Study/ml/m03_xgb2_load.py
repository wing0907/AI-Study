# pkl = python에 특화되어 있음
# joblib = 덩치가 있거나 numpy 또는 가중치 계열 저장할 때는 이걸 씀

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib


# 1. 데이터
x, y = load_breast_cancer(return_X_y=True) #교육용 x y만 따로 빠진다

print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y,
)

# 2. 모델,  #3. 훈련 - 불러오기
path = './_save/m01_job/'
model = XGBClassifier()
model.load_model(path + 'm03_xgb_save.dat')



# 4. 평가, 예측
results = model.score(x_test, y_test) # tensorflow 의  evaluate와 같은 것.
print("최종점수 : ", results)

y_predict = model.predict(x_test)   
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)


# 불러온 점수 동일한지 확인해보기. = 똑같음
# 최종점수 :  0.956140350877193
# accuracy_score : 0.956140350877193



