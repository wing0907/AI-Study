
# https://www.kaggle.com/competitions/nlp-getting-started/data
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, Embedding, LSTM, BatchNormalization, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#  1. data
path = 'C:\Study25\_data\kaggle\\nlp\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
print(train_csv)        # [7613 rows x 4 columns]


def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + 'keyword: ' + df['keyword'] + \
                '   location: ' + df['location']
    return df

train = merge_text(train_csv)
test = merge_text(test_csv)

x1 = train.drop(['target'], axis=1)
y = train['target']
print(x1.shape, y.shape) # (7613, 3) (7613,)

# 수정된 Tokenizer 처리
token = Tokenizer()
token.fit_on_texts(train['text'])
x = token.texts_to_sequences(train['text'])

padding_x = pad_sequences(x, padding='pre', maxlen=150, truncating='pre')  # 길이 여유 있게

x_train, x_test, y_train, y_test = train_test_split(
    padding_x, y, train_size=0.8, shuffle=True, random_state=222)


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))


model = Sequential()
# 수정된 Embedding
vocab_size = len(token.word_index) + 1
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=200, input_length=150))
model.add(Bidirectional(LSTM(units=256)))  # 이 한 줄만으로 충분합니다
model.add(Dense(300, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))



model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

model_path = './_save/keras66/'
filename = f'k66_3_{date}_{{epoch:04d}}-{{val_loss:.4f}}.hdf5'  # 중괄호 이스케이프
filepath = model_path + filename

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()
 
 
loss= model.evaluate(x_test, y_test, verbose=1)

y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score, f1_score

# 예측 및 이진화
y_pred = model.predict(x_test)
y_pred_class = (y_pred > 0.5).astype(int)

# 평가
# 모델 성능 평가
acc = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
print('Accuracy:', round(acc, 4))
print('F1-score:', round(f1, 4))

# Kaggle 제출용 예측
x_test_submit = token.texts_to_sequences(test['text'])
x_test_submit = pad_sequences(x_test_submit, padding='pre', maxlen=150)
y_submit = model.predict(x_test_submit)
y_submit_class = (y_submit > 0.5).astype(int)

# 제출 CSV에 예측값 입력
submission_csv['target'] = y_submit_class
submission_filename = f"C:/Study25/_data/kaggle/nlp/submission_{date}.csv"
submission_csv.to_csv(submission_filename, index=False)
print(f"✅ Submission saved to: {submission_filename}")
# Accuracy: 0.7761
# F1-score: 0.7137
# ✅ Submission saved to: C:/Study25/_data/kaggle/nlp/submission_0630_1729.csv