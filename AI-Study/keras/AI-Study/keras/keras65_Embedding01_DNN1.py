from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

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

# Tokenizing
token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)

# Padding
padding_x = pad_sequences(x, padding='pre', maxlen=5, truncating='pre')

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    padding_x, y, test_size=0.1, random_state=333
)

# 모델 구성
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid'),  # binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 학습
model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.1)

# 평가
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss, '/ acc :', acc)

# 예측
x_pred = ['이삭이 참 잘생겼다']
x_pred_seq = token.texts_to_sequences(x_pred)
x_pred_pad = pad_sequences(x_pred_seq, padding='pre', maxlen=5)
y_pred = model.predict(x_pred_pad)
print('[이삭이 참 잘생겼다]의 결과 :', y_pred)

# 해석 추가
threshold = 0.5
if y_pred[0][0] >= threshold:
    print(">> 예측 결과: 긍정 (확률 {:.2f})".format(y_pred[0][0]))
else:
    print(">> 예측 결과: 부정 (확률 {:.2f})".format(y_pred[0][0]))
    
# loss : 0.5017094612121582 / acc : 0.5
# [이삭이 참 잘생겼다]의 결과 : [[1.]]
# >> 예측 결과: 긍정 (확률 1.00)


# loss : 0.31053426861763 / acc : 1.0
# [이삭이 참 잘생겼다]의 결과 : [[0.99995756]]