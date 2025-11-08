# -*- coding: utf-8 -*-
import os, warnings, math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========== 설정 ==========
DATA_DIR = r"D:\Study25WJ\_data\kaggle\bike"
RANDOM_SEED = 7777
EPOCHS = 500
BATCH_SIZE = 256
LR = 1e-3

# TF1 그래프 모드 + 시드
tf.compat.v1.disable_eager_execution()
np.random.seed(RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

def read_csv_kr(path):
    """한글 경로/인코딩 대응 CSV 로더"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def find_train_csv(data_dir):
    """train.csv 우선, 없으면 폴더 내 단일 CSV 사용"""
    tpath = os.path.join(data_dir, "train.csv")
    if os.path.exists(tpath):
        return tpath
    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if len(csvs) == 1:
        return os.path.join(data_dir, csvs[0])
    # 유사 이름도 탐색
    for name in ["bike_train.csv", "bikesharing_train.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("학습용 CSV를 찾지 못했습니다. (train.csv 권장)")

def add_datetime_parts(df, col="datetime"):
    """datetime 컬럼이 있으면 파생변수 추가"""
    if col in df.columns:
        dt = pd.to_datetime(df[col])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["hour"] = dt.dt.hour
        df["weekday"] = dt.dt.weekday
        # 원본 datetime은 모델 입력에서 제외 (원하면 남겨도 됨)
        df = df.drop(columns=[col])
    return df

def guess_target_column(df):
    """타깃 컬럼 자동: count 우선, 없으면 마지막 유효 열"""
    candidates = ["count","Count","COUNT","target","TARGET","y","Y"]
    for c in candidates:
        if c in df.columns:
            return c
    tail = df.columns[-1]
    if tail.lower() not in {"id","index","idx"}:
        return tail
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (count / target 등 확인)")

def split_cols(df, target_col):
    """숫자/범주 컬럼 분리"""
    known_cats = {"season","holiday","workingday","weather","year","month","hour","weekday","day"}
    cols = [c for c in df.columns if c != target_col]
    # 누수 컬럼 제거 (casual/registered는 train에만 존재, 타깃 count와 강하게 연관됨)
    leak = [c for c in cols if c.lower() in {"casual","registered"}]
    cols = [c for c in cols if c not in leak]
    # object이거나 알려진 범주 → 범주, 나머지 → 숫자
    cat_cols = [c for c in cols if (df[c].dtype == object) or (c.lower() in known_cats)]
    num_cols = [c for c in cols if c not in cat_cols]
    return num_cols, cat_cols, leak

def preprocess_train(df, target_col):
    """훈련용 전처리: 누수 제거, datetime 파생, 원-핫/스케일"""
    # ID/인덱스 제거
    drop_like = {"id","index","idx","일련번호","번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # datetime 파생
    df = add_datetime_parts(df, col="datetime")

    # 타깃
    y = df[target_col].astype(np.float32).values.reshape(-1,1)

    # 숫자/범주 분리 + 누수 제거
    num_cols, cat_cols, leak = split_cols(df, target_col)
    if leak:
        df = df.drop(columns=leak)

    # 숫자 처리
    Xn = df[num_cols].copy()
    for c in num_cols:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    Xn = Xn.fillna(Xn.median(numeric_only=True))

    # 범주 처리 (원-핫)
    if cat_cols:
        Xc = pd.get_dummies(df[cat_cols].astype(str).fillna("NA"), drop_first=False)
    else:
        Xc = pd.DataFrame(index=df.index)

    # 스케일러 (숫자만)
    scaler = StandardScaler()
    if len(num_cols) > 0:
        Xn_scaled = pd.DataFrame(
            scaler.fit_transform(Xn), columns=num_cols, index=df.index
        ).astype(np.float32)
    else:
        Xn_scaled = Xn

    # 결합
    X_all = pd.concat([Xn_scaled, Xc.astype(np.float32)], axis=1)
    feature_names = X_all.columns.tolist()
    return X_all.values.astype(np.float32), y, scaler, feature_names

def preprocess_infer(df, target_col, scaler, feature_names):
    """추론/평가 전처리: 학습 스키마에 맞춰 정렬/결측 0 채움"""
    drop_like = {"id","index","idx","일련번호","번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = add_datetime_parts(df, col="datetime")

    y = None
    if target_col in df.columns:
        y = df[target_col].astype(np.float32).values.reshape(-1,1)

    num_cols, cat_cols, leak = split_cols(df, target_col)
    if leak:
        df = df.drop(columns=leak)

    Xn = df[num_cols].copy()
    for c in num_cols:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    Xn = Xn.fillna(Xn.median(numeric_only=True))

    if cat_cols:
        Xc = pd.get_dummies(df[cat_cols].astype(str).fillna("NA"), drop_first=False)
    else:
        Xc = pd.DataFrame(index=df.index)

    if len(num_cols) > 0:
        Xn_scaled = pd.DataFrame(
            scaler.transform(Xn), columns=num_cols, index=df.index
        ).astype(np.float32)
    else:
        Xn_scaled = Xn

    X_all = pd.concat([Xn_scaled, Xc.astype(np.float32)], axis=1)
    X_all = X_all.reindex(columns=feature_names, fill_value=0.0)
    return X_all.values.astype(np.float32), y

# ========== 데이터 로드 & 전처리 ==========
train_path = find_train_csv(DATA_DIR)
df_train = read_csv_kr(train_path)
target = guess_target_column(df_train)
print(f"[INFO] 사용 파일: {train_path}")
print(f"[INFO] 타깃 컬럼: {target}")
print(f"[INFO] 열(앞부분): {list(df_train.columns)[:12]} ...")

X_all, y_all, scaler, feat_names = preprocess_train(df_train, target)

# 학습/검증 분리
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_SEED
)
n_features = X_tr.shape[1]

# ========== TF1 MLP (ReLU + Dropout) ==========
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 드롭아웃 제어 (훈련: 0.9, 평가: 1.0)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

he = tf.compat.v1.keras.initializers.he_uniform()

# n_features -> 256 -> 128 -> 1
w1 = tf.compat.v1.Variable(he(shape=[n_features, 256])); b1 = tf.compat.v1.Variable(tf.zeros([256]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)

w2 = tf.compat.v1.Variable(he(shape=[256, 128])); b2 = tf.compat.v1.Variable(tf.zeros([128]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(he(shape=[128, 1])); b3 = tf.compat.v1.Variable(tf.zeros([1]))
pred = tf.matmul(h2, w3) + b3  # 회귀: 선형 출력

# 손실/옵티마이저
mse = tf.reduce_mean(tf.square(pred - yph))
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(mse)

# 학습 루프
steps_per_epoch = int(math.ceil(X_tr.shape[0] / BATCH_SIZE))
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, EPOCHS + 1):
        # 셔플
        idx = np.random.permutation(X_tr.shape[0])
        Xb = X_tr[idx]; yb = y_tr[idx]

        for step in range(steps_per_epoch):
            s = step * BATCH_SIZE
            e = min((step + 1) * BATCH_SIZE, X_tr.shape[0])
            feed = {
                x: Xb[s:e], yph: yb[s:e],
                keep_prob1: 0.9, keep_prob2: 0.9  # ▶ 훈련 시 드롭아웃 ON
            }
            sess.run(train_op, feed_dict=feed)

        if epoch % 25 == 0 or epoch == 1:
            tr_mse = sess.run(mse, feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0})
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train MSE: {tr_mse:.4f}")

    # 평가 (드롭아웃 OFF)
    y_tr_pred = sess.run(pred, feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0})
    y_te_pred = sess.run(pred, feed_dict={x: X_te, yph: y_te, keep_prob1: 1.0, keep_prob2: 1.0})

# 지표
from math import sqrt
tr_mse = mean_squared_error(y_tr, y_tr_pred); te_mse = mean_squared_error(y_te, y_te_pred)
tr_rmse, te_rmse = sqrt(tr_mse), sqrt(te_mse)
tr_mae = mean_absolute_error(y_tr, y_tr_pred); te_mae = mean_absolute_error(y_te, y_te_pred)
tr_r2  = r2_score(y_tr, y_tr_pred);          te_r2  = r2_score(y_te, y_te_pred)

print("\n=== Results (Kaggle Bike / ReLU + Dropout MLP) ===")
print(f"Train RMSE: {tr_rmse:.3f} | MAE: {tr_mae:.3f} | R²: {tr_r2:.3f}")
print(f" Test RMSE: {te_rmse:.3f} | MAE: {te_mae:.3f} | R²: {te_r2:.3f}")

# ================== (선택) test.csv 예측 & 제출 저장 — 요청에 따라 주석 처리 ==================
# test_path = os.path.join(DATA_DIR, "test.csv")
# if os.path.exists(test_path):
#     df_test = read_csv_kr(test_path)
#     # test에는 count가 없으므로 target_col은 더미로 넘김
#     X_submit, _ = preprocess_infer(df_test.assign(count=0), target, scaler, feat_names)
#     # with tf.compat.v1.Session() as sess:
#     #     # 실제 제출용이라면 학습된 가중치를 저장/복원(tf.train.Saver)해야 합니다.
#     #     # 여기서는 예시로만 남깁니다.
#     #     pass
#     # y_pred_submit = ...  # 학습 모델로 X_submit 예측
#     # sub = pd.DataFrame({
#     #     "datetime": df_test["datetime"] if "datetime" in df_test.columns else np.arange(len(df_test)),
#     #     "count": y_pred_submit.ravel()
#     # })
#     # sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False, encoding="utf-8-sig")
#     # print("[INFO] submission.csv 저장 완료")

# === Results (Kaggle Bike / ReLU + Dropout MLP) ===
# Train RMSE: 11.145 | MAE: 7.900 | R²: 0.996
#  Test RMSE: 35.676 | MAE: 22.193 | R²: 0.961