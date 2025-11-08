import numpy as np
import pandas as pd
from sklearn.datasets import load_iris          # 다중분류
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time
# 1. 데이터
datasets = load_iris()
from tensorflow.keras.utils import to_categorical


x = datasets.data
y = datasets.target
print(x.shape, y.shape)                         # (150, 4) (150,)

print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
print(np.unique(y, return_counts=True))         # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

print(pd.DataFrame(y).value_counts())
# 0    50
# 1    50
# 2    50
print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

# y는 클래스 번호(정수)로 된 레이블 배열이라고 가정
y_onehot = to_categorical(y)  # (샘플수, 클래스수) 형태로 변환
print(y_onehot)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=333,
)

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

#  2. 모델구성                                  # 다중분류에서는 "반드시" OneHotEncoding을 사용하고 Y 만 적용한다!!!!!!!!
from tensorflow.keras.layers import BatchNormalization

num_classes = y_onehot.shape[1]

model = Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

#  3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=100,
    restore_best_weights=True
)

start_time = time.time()
model.fit(x_train, y_train_onehot, validation_data=(x_test, y_test_onehot),
        epochs=1000, batch_size=8, verbose=1,  # validation_split=0.1,
        callbacks=[es])
end_time = time.time()



#  4. 평가, 예측
loss = model.evaluate(x_test, y_test_onehot)
print('loss:', loss[0])
print('acc:', loss[1])



# loss: 0.04939129203557968
# acc: 1.0

# loss: 0.08271361887454987 # batchnormalization 뺐을 때의 값
# acc: 1.0
