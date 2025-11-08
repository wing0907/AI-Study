from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
print(padding_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
print(padding_x.shape)  # (15, 5)


#  2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding

# model = Sequential()
# model.add(Embedding(input_dim=10, output_dim=5, input_length=3))
# model.add(Dense(1))
# model.summary()
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 3, 5)              50

#  dense (Dense)               (None, 3, 1)              6

# =================================================================
# Total params: 56
# Trainable params: 56
# Non-trainable params: 0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=100, input_length=5, ))
# model.add(LSTM(16))
# model.add(Dense(1))
# model.summary()
# 31개의 단어(단어사전의 개수)를 100개의 백터(나가는 차원의 개수) 형태인 5개의 시퀀스로 출력한다 

# input_dim= 단어사전의 개수(말뭉치의 개수) 고정값
# output_dim= 다음 레이어로 전달하는 노드의 개수. 튜닝가능 (예.units= filters=)
# input_length= 시퀀스의 개수 또는 컬럼의 개수. 고정값
#               (N, 5), 컬럼의 개수, 문장의 시퀀스의 개수.

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, 5, 100)            3100

#  lstm (LSTM)                 (None, 16)                7488

#  dense (Dense)               (None, 1)                 17

# =================================================================
# Total params: 10,605
# Trainable params: 10,605
# Non-trainable params: 0



# model = Sequential()
# model.add(Embedding(input_dim=31, output_dim=100, ))
# model.add(LSTM(16))
# model.add(Dense(1))
# model.summary()
# input_length는 생략해도 알아서 맞춰줌. // 모르면 짜지고 알면 정확하게 기입해라 이 ㅂㅅ아! 
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  embedding (Embedding)       (None, None, 100)         3100

#  lstm (LSTM)                 (None, 16)                7488

#  dense (Dense)               (None, 1)                 17

# =================================================================
# Total params: 10,605
# Trainable params: 10,605
# Non-trainable params: 0



# model = Sequential()
# model.add(Embedding(input_dim=13, output_dim=100, ))
# model.add(LSTM(16))
# model.add(Dense(1))
# model.summary()
#  input_dim= 기존 단어사전 개수 보다 많으면 그만큼의 데이터 소모량이 커짐
#  input_dim= 기존 단어사전 개수 보다 적으면 그만큼의 데이터 소모량이 작아지고 성능도 떨어짐

#  input_dim=13 결과
# loss/acc :  [0.731265127658844, 0.6000000238418579]
# [[ 0  0 28  1 27]]
# 이삭이 참 잘생겼다 의 결과 :  [[0.]]

#  input_dim 이 31개 인 이유는 padding 때문에 + 1 이 됨.




model = Sequential()
# model.add(Embedding(13, 100))       # 돌아감
# model.add(Embedding(13, 100, 5 )) # 이래하면 ValueError 뜸
model.add(Embedding(13, 100, input_length=5 )) # 돌아감 1 해도 돌아가지만 굳이굳이. 구지구지
model.add(LSTM(16))
model.add(Dense(1))
model.summary()





#  3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(padding_x, labels, epochs=100,)

#  4. 평가, 예측
results = model.evaluate(padding_x, labels)
print('loss/acc : ', results)

x_pred = ['이삭이 참 잘생겼다']
x_pred = token.texts_to_sequences(x_pred)
x_pred = pad_sequences(x_pred, maxlen=5)
print(x_pred)

y_pred = model.predict(x_pred)
print('이삭이 참 잘생겼다 의 결과 : ', np.round(y_pred))

# loss/acc :  [0.02125471644103527, 1.0]
# [[ 0  0 28  1 27]]
# 이삭이 참 잘생겼다 의 결과 :  [[1.]]

