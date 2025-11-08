import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Conv1D, GlobalMaxPooling1D, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import lightgbm as lgb
import warnings
import datetime
warnings.filterwarnings('ignore')



# 1. 데이터 로딩 및 전처리
path = 'C:/Study25/_data/kaggle/nlp/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv['keyword'] = train_csv['keyword'].fillna('unknown')
train_csv['location'] = train_csv['location'].fillna('unknown').str.lower()
test_csv['keyword'] = test_csv['keyword'].fillna('unknown')
test_csv['location'] = test_csv['location'].fillna('unknown').str.lower()

x_text_raw = train_csv['text']
x_keyword = train_csv[['keyword']]
x_location = train_csv[['location']]
y = train_csv['target'].values

# Tokenizer + padding (for text sequence)
token = Tokenizer(num_words=10000)
token.fit_on_texts(x_text_raw)
x_seq = token.texts_to_sequences(x_text_raw)
x_seq = pad_sequences(x_seq, maxlen=100, padding='post')

x_text_test_raw = test_csv['text']
x_seq_test_submission = token.texts_to_sequences(x_text_test_raw)
x_seq_test_submission = pad_sequences(x_seq_test_submission, maxlen=100, padding='post')

# One-Hot Encoding for keyword/location
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
x_cat = ohe.fit_transform(pd.concat([x_keyword, x_location], axis=1))
num_kw = train_csv['keyword'].nunique()
x_keyword_ohe = x_cat[:, :num_kw]
x_location_ohe = x_cat[:, num_kw:]

test_cat = ohe.transform(pd.concat([test_csv[['keyword']], test_csv[['location']]], axis=1))
x_keyword_test = test_cat[:, :num_kw]
x_location_test = test_cat[:, num_kw:]

# TF-IDF (for LightGBM)
tfidf = TfidfVectorizer(max_features=3000)
x_tfidf = tfidf.fit_transform(x_text_raw)
x_tfidf_submission = tfidf.transform(x_text_test_raw)

# Train/Test Split
x_seq_train, x_seq_test, x_kw_train, x_kw_test, x_loc_train, x_loc_test, x_tfidf_train, x_tfidf_test, y_train, y_test = train_test_split(
    x_seq, x_keyword_ohe, x_location_ohe, x_tfidf, y, train_size=0.85, random_state=222, stratify=y)

# 2. 모델 정의 (LSTM + keyword + location 병합)
vocab_size = min(10000, len(token.word_index) + 1)

input_text = Input(shape=(100,))
x = Embedding(vocab_size, 64)(input_text)
x = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))(x)

input_kw = Input(shape=(x_keyword_ohe.shape[1],))
kw_dense = Dense(16, activation='relu')(input_kw)

input_loc = Input(shape=(x_location_ohe.shape[1],))
loc_dense = Dense(16, activation='relu')(input_loc)

merged = Concatenate()([x, kw_dense, loc_dense])
dense = Dense(32, activation='relu')(merged)
dense = Dropout(0.2)(dense)
output = Dense(1, activation='sigmoid')(dense)

lstm_model = Model(inputs=[input_text, input_kw, input_loc], outputs=output)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# CNN 모델은 그대로 유지
def build_cnn_model():
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=100))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

cnn_model = build_cnn_model()
lgbm = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.07)

# 3. 학습
print("Training LSTM (with keyword/location)...")
lstm_model.fit([x_seq_train, x_kw_train, x_loc_train], y_train, epochs=3, batch_size=64, verbose=0)

print("Training CNN...")
cnn_model.fit(x_seq_train, y_train, epochs=3, batch_size=64, verbose=0)

print("Training LightGBM...")
lgbm.fit(x_tfidf_train, y_train)

# 4. 예측
pred_lstm = lstm_model.predict([x_seq_test, x_kw_test, x_loc_test]).flatten()
pred_cnn = cnn_model.predict(x_seq_test).flatten()
pred_lgbm = lgbm.predict_proba(x_tfidf_test)[:, 1]

# 5. 앙상블 (soft voting with weights)
weighted_avg = (0.4 * pred_lstm + 0.3 * pred_cnn + 0.3 * pred_lgbm)
pred_final = (weighted_avg > 0.5).astype(int)

# 6. 평가
acc = accuracy_score(y_test, pred_final)
f1 = f1_score(y_test, pred_final)

print(f"\n✅ 앙상블 Accuracy: {acc:.4f}")
print(f"✅ 앙상블 F1-score: {f1:.4f}")

# 7. 제출용 예측
submit_pred_lstm = lstm_model.predict([x_seq_test_submission, x_keyword_test, x_location_test]).flatten()
submit_pred_cnn = cnn_model.predict(x_seq_test_submission).flatten()
submit_pred_lgbm = lgbm.predict_proba(x_tfidf_submission)[:, 1]

submit_weighted_avg = (0.4 * submit_pred_lstm + 0.3 * submit_pred_cnn + 0.3 * submit_pred_lgbm)
submission_csv['target'] = (submit_weighted_avg > 0.5).astype(int)

# 8. 저장
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission_path = f'C:/Study25/_data/kaggle/nlp/submission_{date}.csv'
submission_csv.to_csv(submission_path, index=False)

# 모델 가중치 저장
lstm_model.save(f'C:/Study25/_data/kaggle/nlp/lstm_model_{date}.h5')
cnn_model.save(f'C:/Study25/_data/kaggle/nlp/cnn_model_{date}.h5')

print(f"\n📁 Submission saved to: {submission_path}")
print(f"💾 LSTM & CNN model weights saved with timestamp {date}")


# ✅ 앙상블 Accuracy: 0.8039            Score: 0.79037
# ✅ 앙상블 F1-score: 0.7691

# 📁 Submission saved to: C:/Study25/_data/kaggle/nlp/submission_0630_1858.csv
# 💾 LSTM & CNN model weights saved with timestamp 0630_1858


# ✅ 앙상블 Accuracy: 0.8266            Score: 0.78761
# ✅ 앙상블 F1-score: 0.7916

# 📁 Submission saved to: C:/Study25/_data/kaggle/nlp/submission_0701_0947.csv
# 💾 LSTM & CNN model weights saved with timestamp 0701_0947