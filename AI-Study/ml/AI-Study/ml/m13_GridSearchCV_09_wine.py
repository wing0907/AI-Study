import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time

# 1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets['target']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8,
    stratify=y
)


n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {"C":[1,10,100,1000], "kernel":['linear', 'sigmoid'],
        'degree':[3,4,5]},                                          # 24번
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001,0.0001]},     # 6번
    {'C':[1,10,100,1000], 'kernel':['sigmoid'],
      'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}                  # 24번
]                                                                   # 총 54번

# 2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold,       # 54 * 5 = 270번
                     verbose=1,
                     refit=True, # 다시 훈련                         # 1번
                     n_jobs=-1, # CPU를 FULL로 돌리겠다라는 뜻
                     )                                              # 총 271번

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수 : ', model.best_estimator_) # 최적의 파라미터가 출력 됨 (지정한 것 중에 제일 좋은 것)
print('최적의 매개변수 : ', model.best_params_) # 최적의 파라미터가 출력 됨 (전체 중에 제일 좋은 것)

# 4. 평가, 예측
print('best_score : ', model.best_score_)    # train에서 가장 좋은 score 가 나옴

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))
print('time : ', round(end - start, 4), 'seconds')

# 최적의 매개변수 :  SVC(C=10, kernel='linear')
# 최적의 매개변수 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
# best_score :  0.9916666666666668
# model.score :  0.9
# accuracy_score :  0.9
# time :  1.907 seconds