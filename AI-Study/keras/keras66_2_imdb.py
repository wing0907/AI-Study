from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000,
)
print(x_train)
print(y_train)  # [1 0 0 ... 0 1 0] binary
print(x_train.shape, y_train.shape)     # (25000,) (25000,) x의 (25000,)개의 리스트 // y는 순수한 벡터 (25000,)개
print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) 라벨이 균형데이터 acc 사용하면됨. 불균형이어도 상관없음
print(pd.value_counts(y_train))
# 1    12500
# 0    12500
# dtype: int64

print("영화평의 최대길이 : ", max(len(i) for i in x_train))         # 2494
print("영화평의 최소길이 : ", min(len(i) for i in x_train))         # 11
print("영화평의 평균길이 : ", sum(map(len, x_train))/len(x_train))  # 238.71364

# 실습 - 만들어보기 acc > 0.85

maxlen = 1000
x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=128, input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es])

# 6. 평가 및 예측
results = model.evaluate(x_test, y_test)
print("loss/acc:", results)

# 7. 예측
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("예측 결과 : ", y_pred_classes)

# loss/acc: [0.3267871141433716, 0.8712000250816345]