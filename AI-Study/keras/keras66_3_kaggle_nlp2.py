import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Bidirectional, Concatenate
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
from sklearn.utils import class_weight

# ===============================
# 1. Load Data & Preprocess
# ===============================
path = 'C:/Study25/_data/kaggle/nlp/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

def preprocess_columns(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    return df

train = preprocess_columns(train_csv)
test = preprocess_columns(test_csv)

# ✨ 컬럼 분리
train_text = train['text']
train_keyword = train[['keyword']]
train_location = train[['location']]
test_text = test['text']
test_keyword = test[['keyword']]
test_location = test[['location']]
y = train['target'].values

# ===============================
# 2. Tokenizer & OneHot Encoding
# ===============================
# Text 처리
token = Tokenizer()
token.fit_on_texts(train_text)
x_text = token.texts_to_sequences(train_text)
x_text = pad_sequences(x_text, padding='pre', maxlen=150)

# Categorical 처리
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
x_cat = ohe.fit_transform(pd.concat([train_keyword, train_location], axis=1))

# Keyword/Location 분리
num_kw = train['keyword'].nunique()
x_keyword = x_cat[:, :num_kw]
x_location = x_cat[:, num_kw:]

# ===============================
# 3. Train-Test Split
# ===============================
x_text_train, x_text_test, x_kw_train, x_kw_test, x_loc_train, x_loc_test, y_train, y_test = train_test_split(
    x_text, x_keyword, x_location, y, train_size=0.8, shuffle=True, random_state=222)


# 3개의 입력을 하나로 합칩니다 (sklearn 모델용)
x_train_all = np.concatenate([x_text_train, x_kw_train, x_loc_train], axis=1)
x_test_all  = np.concatenate([x_text_test,  x_kw_test,  x_loc_test],  axis=1)

max_name = ""
max_score = 0
allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train_all, y_train)
        results = model.score(x_test_all, y_test)
        print(name, '의 정답률 :', results)

        if results > max_score:
            max_score = results
            max_name = name
    except Exception as e:
        print(name, "은(는) 에러뜬 분!!!", str(e))

print("============================================")
print("최고모델 :", max_name, max_score)
print("============================================")

# 7. 제출파일 생성
# ===============================
# test 데이터 동일 처리
x_text_sub = token.texts_to_sequences(test_text)
x_text_sub = pad_sequences(x_text_sub, padding='pre', maxlen=150)

x_cat_sub = ohe.transform(pd.concat([test_keyword, test_location], axis=1))
x_kw_sub = x_cat_sub[:, :num_kw]
x_loc_sub = x_cat_sub[:, num_kw:]

# 예측 및 저장
x_all_sub = np.concatenate([x_text_sub, x_kw_sub, x_loc_sub], axis=1)
y_submit = model.predict(x_all_sub)  # 마지막 학습된 모델 기준

submission_csv['target'] = (y_submit > 0.5).astype(int)

date = datetime.datetime.now().strftime("%m%d_%H%M")
sub_path = f'C:/Study25/_data/kaggle/nlp/submission_{date}.csv'
submission_csv.to_csv(sub_path, index=False)
print(f"📁 Submission saved to: {sub_path}")

# ✅ Accuracy: 0.7919, F1-score: 0.7408, Time: 23.38 sec
# 📁 Submission saved to: C:/Study25/_data/kaggle/nlp/submission_0630_1807.csv

# ExtraTreesClassifier 0.7432698621142482