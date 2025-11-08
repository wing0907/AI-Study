import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC       # svm support vector machine
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 0, 0, 1]

# 2. 모델
# model = Perceptron()
# model = LogisticRegression()  # 0.75 나옴 나머지는 1.0나옴
model = LinearSVC()



# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print("model.score :" , results)

acc = accuracy_score(y_data, y_predict)
print("acc :" , acc)


# model.score : 1.0
# acc : 1.0