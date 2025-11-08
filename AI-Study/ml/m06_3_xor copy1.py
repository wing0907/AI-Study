import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 원본 데이터
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])  # XOR

# 2. 새로운 특성 추가: AND (비선형)
and_feature = (x[:,0] & x[:,1]).reshape(-1, 1)
x_new = np.concatenate([x, and_feature], axis=1)

# 3. 모델 적용
model = LinearSVC()
# model = Perceptron()
model.fit(x_new, y)

y_pred = model.predict(x_new)
print("acc:", accuracy_score(y, y_pred))


# acc: 1.0