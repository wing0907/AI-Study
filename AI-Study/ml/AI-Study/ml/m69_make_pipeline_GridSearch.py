import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline    # make_pipeline는 함수, Pipeline 클래스 
from sklearn.model_selection import GridSearchCV


# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=777,
    stratify=y
)

parameters = [                          # 딕셔너리형태
    {'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [5, 6, 10], 'randomforestclassifier__min_samples_leaf' : [3, 10]}, # 2 x 3 x 2 = 12번 돌아감
    {'randomforestclassifier__max_depth' : [6,8,10,12], 'randomforestclassifier__min_samples_leaf' : [3,5,7,10]},                          # 4 x 4 = 16
    {'randomforestclassifier__min_samples_leaf' : [3,5,7,9], 'randomforestclassifier__min_samples_leaf' : [2,3,5,10]},                      # 4 x 4 = 16
    {'randomforestclassifier__min_samples_split' : [2,3,5,6]},        
] # make_pipeline으로 구성하려면 full name을 넣어줘야 함.
  # 'randomforestclassifier__+... 이런식, 어차피 성능은 똑같아서 길이 생각해서 Pipeline을 사용 권장함.

# # 2. 모델
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
# pipe = Pipeline([('std', MinMaxScaler()), ('rf', RandomForestClassifier())])

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)  # cross validation


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score:', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score:', acc)



