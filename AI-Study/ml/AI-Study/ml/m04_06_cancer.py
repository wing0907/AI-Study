from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

#  1.데이터
datasets = load_breast_cancer
x, y = load_breast_cancer(return_X_y=True)

print(x)
print(y)
print(x.shape, y.shape)

## 최종 결과 예시 ==> ##
# LinearSVC : 0.7
# LogisticRegression : 0.8
# DecisionTreeClassifier : 0.9
# RandomForestClassifier : 1.0

# model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, 
#               RandomForestClassifier,]

# 2. 모델 리스트 (이름과 클래스 쌍으로 저장)
model_list = [
    ('LinearSVC', LinearSVC),
    ('LogisticRegression', LogisticRegression),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]
# for i in model_list:
#     print(i)

# 3. 모델 반복 학습 및 결과 출력
for name, model_class in model_list:
    model = model_class()  # 인스턴스 생성
    model.fit(x, y)
    score = model.score(x, y)
    print(f'{name} : {score:.4f}')
    

model.fit(x, y)

results = model.score(x, y)  # 이거 쓰면 끝
print(results)

# LinearSVC : 0.9315
# LogisticRegression : 0.9473
# DecisionTreeClassifier : 1.0000
# RandomForestClassifier : 1.0000

