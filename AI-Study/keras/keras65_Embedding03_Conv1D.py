import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
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

# 2. 토큰화
token = Tokenizer()
token.fit_on_texts(docs)
x_seq = token.texts_to_sequences(docs)

# 3. 패딩
maxlen = 5
x_pad = pad_sequences(x_seq, padding='pre', maxlen=maxlen)

# 4. get_dummies로 원핫 인코딩
vocab_size = len(token.word_index) + 1
x_onehot = []
for sentence in x_pad:
    df = pd.get_dummies(sentence).reindex(columns=range(vocab_size), fill_value=0)
    x_onehot.append(df.values)

x_onehot = np.array(x_onehot)  # shape: (15, 5, vocab_size)

# 5. train/test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_onehot, y, test_size=0.3, random_state=222
)

# 6. Conv1D + Flatten 모델 구성 (Embedding 제거됨!)
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(maxlen, vocab_size)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류

# 7. 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=4, validation_split=0.2)

# 8. 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss:', loss, 'acc:', acc)

# 9. 예측
x_pred = ['이삭이 참 잘생겼다']
x_pred_seq = token.texts_to_sequences(x_pred)
x_pred_pad = pad_sequences(x_pred_seq, padding='pre', maxlen=maxlen)

# 원핫 변환
x_pred_onehot = []
for sentence in x_pred_pad:
    df = pd.get_dummies(sentence).reindex(columns=range(vocab_size), fill_value=0)
    x_pred_onehot.append(df.values)
x_pred_onehot = np.array(x_pred_onehot)

# 예측
y_pred = model.predict(x_pred_onehot)
print('[이삭이 참 잘생겼다]의 결과 :', y_pred)

# 해석 추가
threshold = 0.5
if y_pred[0][0] >= threshold:
    print(">> 예측 결과: 긍정 (확률 {:.2f})".format(y_pred[0][0]))
else:
    print(">> 예측 결과: 부정 (확률 {:.2f})".format(y_pred[0][0]))


# loss: 0.7641973495483398 acc: 0.4000000059604645
# [이삭이 참 잘생겼다]의 결과 : [[0.652949]]
# >> 예측 결과: 긍정 (확률 0.65)