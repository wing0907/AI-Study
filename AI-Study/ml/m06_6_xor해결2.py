import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC # svm support vector machine
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = Perceptron()        # 0.5
# model = LinearSVC()           # 0.5
# model = SVC()       # 바아아아로 1.0 나오쥬~~?
model = DecisionTreeClassifier()        # 얘도 바아아아로 1.0 나오쥬~~??


# 3. 컴파일, 훈련
model.fit(x_data, y_data)


# 4. 평가, 예측
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print("model.score :" , results)

acc = accuracy_score(y_data, np.round(y_predict))
print("acc :" , acc)

# model.score : [0.21986936032772064, 1.0]
# acc : 1.0