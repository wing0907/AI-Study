# pkl = python에 특화되어 있음
# joblib = 덩치가 있거나 numpy 또는 가중치 계열 저장할 때는 이걸 씀

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True) #교육용 x y만 따로 빠진다

print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y,
)

parameters = {'n_estimators' : 1000,
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 1,
              'min_child_weight' : 1,
              'subsample' : 1,
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,
              'reg_lambda' : 1,
              'random_state' : 3377,
            #   'verbose' : 0,  이미 적용되어 있음
              }

# 2. 모델
model = XGBClassifier(
    # **parameters,               # * 1개 ** 2개 내용정리 해서 이메일로 보낼 것. 0630과제
)


# 3. 훈련
model.set_params(**parameters,
                 early_stopping_rounds=10,)   # parameters만 fit 전에 불러와주기 


model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],       # early_stopping_rounds 하려면 eval_sets 써야함.
          verbose=10,)


# 4. 평가, 예측
results = model.score(x_test, y_test) # tensorflow 의  evaluate와 같은 것.
print("최종점수 : ", results)
# 최종점수 :  0.9473684210526315

y_predict = model.predict(x_test)   
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)
# accuracy_score : 0.9473684210526315

#  verbose가 적용되고 있다. verbose 없애면 아래 에러 사라짐
# Parameters: { "verbose" } are not used.
#   warnings.warn(smsg, UserWarning)

# **parameters 사용 후 올라감
# 최종점수 :  0.956140350877193
# accuracy_score : 0.956140350877193

path = './_save/m01_job/'
import joblib
joblib.dump(model, path + 'm01_joblib_save.joblib')

