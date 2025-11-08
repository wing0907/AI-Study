from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 불러오기
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000, # 단어사전의 개수, 빈도수가 높은 단어 순으로 1000개 뽑는다.
    test_split=0.2, 
    # maxlen=200,     # 단어 길이가 200개까지 있는 문장. 
)

print(x_train)
print(x_train.shape, x_test.shape)      # (8982,) (2246,)
print(y_train.shape, y_test.shape)      # (8982,) (2246,)
print(y_train[0])                       # 3
print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
#  총 46개

#  (8982, 100)으로 바꾸기


print(type(x_train))        # <class 'numpy.ndarray'>
print(type(x_train[0]))     # <class 'list'>

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)  )
print("뉴스기사의 최소길이 : ", min(len(i) for i in x_train))
print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/len(x_train))
# 뉴스기사의 최대길이 :  2376
# 뉴스기사의 최소길이 :  13
# 뉴스기사의 평균길이 :  145.5398574927633


# 2. 시퀀스 패딩
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='pre')

# 3. 레이블 원핫 (or 주석처리해서 sparse 사용해도 됨)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=128, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(46, activation='softmax'))

# 5. 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

model.fit(x_train, y_train, epochs=90, batch_size=16, validation_split=0.1, callbacks=[es])

# 6. 평가 및 예측
results = model.evaluate(x_test, y_test)
print("loss/acc:", results)

# 7. 예측
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("46개의 예측 결과 : ", y_pred_classes[:46])

# loss/acc: [1.2341196537017822, 0.699020504951477]
# 46개의 예측 결과 :  [ 3 10 19 25 16  3  3  3  3  3  1  4  1  3 13  3 25  3 19  3 19  3  3  4
#   9  3  4 25 10  3  3 10  4  3 19  4 19  1  4  3  1 11  3 16  4  4]