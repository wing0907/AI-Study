import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
docs = [
    '너무 재미있다', '참 최고에요', '참 잘만든 영화에요',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요', '생각보다 지루해요', '연기가 어색해요',
    '재미없어요', '너무 재미없다.', '참 재밌네요.',
    '석준이 바보', '준희 잘생겼다', '이삭이 또 구라친다',
]
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])
y = labels

# 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

# 패딩
maxlen = 5
x = pad_sequences(x, padding='pre', maxlen=maxlen)

# 단어 사전 크기 (vocab_size)
vocab_size = len(token.word_index) + 1  # 1 더해주는 이유: padding(0) 포함

# train/test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=222
)

# 2. 모델 구성 (Embedding + LSTM)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류

# 3. 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=4, validation_split=0.2)

# 4. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss, 'acc:', acc)

# 5. 예측
x_pred = ['이삭이 참 잘생겼다']
x_pred_seq = token.texts_to_sequences(x_pred)
x_pred_pad = pad_sequences(x_pred_seq, padding='pre', maxlen=maxlen)
y_pred = model.predict(x_pred_pad)
print('[이삭이 참 잘생겼다]의 결과 :', y_pred)

# 해석 추가
threshold = 0.5
if y_pred[0][0] >= threshold:
    print(">> 예측 결과: 긍정 (확률 {:.2f})".format(y_pred[0][0]))
else:
    print(">> 예측 결과: 부정 (확률 {:.2f})".format(y_pred[0][0]))

# loss: 0.9215987920761108 acc: 0.800000011920929
# [이삭이 참 잘생겼다]의 결과 : [[0.14734212]]
# >> 예측 결과: 부정 (확률 0.15)
