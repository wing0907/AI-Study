from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_digits, load_wine

# 1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier(),]

# for index, value in enumerate(list):
#     print(index, value)
#  enumerate 적용해서 돌려보기. 한번에 4개의 데이터를 4개의 모델로 16번 돌려보리기~
