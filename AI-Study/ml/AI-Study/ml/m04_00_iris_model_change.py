import numpy as np
from sklearn.datasets import load_iris


# 1. 데이터
datasets = load_iris()
# x = datasets.data
# y = datasets['target']      # 이렇게도 쓸 수 있음 (보통 datasets.target 했었음)
x, y = load_iris(return_X_y=True) # 이렇게도 쓸 수 있음 (교육용)
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,)

# 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

# model = LinearSVC(C=0.3)        # c라는 파라미터는 뭔지 모름. 위에 모델을 이 한줄로 요약됨

# model = LogisticRegression()      # 0.97 나옴
# model = DecisionTreeClassifier()    # 1.0 나옴
model = RandomForestClassifier()      # 1.0 나옴  

# 3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy',   # 이거 쓰면 원핫 안해도 됨
#               optimizer='adam',
#               metrics=['acc'],
#               )

# model.fit(x, y, epochs=100)
model.fit(x, y)         # 이거 쓰면 끝

# 4. 평가, 예측
# results = model.evaluate(x, y)
results = model.score(x, y)  # 이거 쓰면 끝
print(results)
