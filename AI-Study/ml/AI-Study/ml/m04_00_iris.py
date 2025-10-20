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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',   # 이거 쓰면 원핫 안해도 됨
              optimizer='adam',
              metrics=['acc'],
              )

model.fit(x, y, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x, y)

print(results)
