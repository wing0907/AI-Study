# -*- coding: utf-8 -*-
import os, warnings, math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ===== 설정 =====
DATA_DIR = r"D:\Study25WJ\_data\kaggle\bank"
RANDOM_SEED = 7777
EPOCHS = 600
BATCH_SIZE = 256
LR = 1e-3

# TF1 그래프 모드 + 시드
tf.compat.v1.disable_eager_execution()
np.random.seed(RANDOM_SEED)
tf.compat.v1.set_random_seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

# ---------- 파일 로드 ----------
def read_csv_kr(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def load_bank_csv(data_dir):
    # 우선순위: train.csv → bank.csv → bank-full.csv → bank_marketing.csv → 폴더 내 유일 CSV
    candidates = ["train.csv", "bank.csv", "bank-full.csv", "bank_full.csv", "bank_marketing.csv", "bankmarketing.csv"]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"[INFO] 사용 파일: {p}")
            return read_csv_kr(p)
    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if len(csvs) == 1:
        p = os.path.join(data_dir, csvs[0])
        print(f"[INFO] 사용 파일(자동): {p}")
        return read_csv_kr(p)
    raise FileNotFoundError("학습용 CSV를 찾지 못했습니다.")

# ---------- 전처리 ----------
def guess_target_column(df):
    # 흔한 타깃 이름 후보(yes/no)
    for c in ["y","deposit","subscribed","subscription","target","TARGET","label","Label","class","Class"]:
        if c in df.columns:
            return c
    # 없다면 고유값 2개(이진)인 컬럼을 뒤에서 앞으로 탐색
    for c in df.columns[::-1]:
        if df[c].nunique(dropna=True) == 2:
            return c
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (y/deposit/target 등 확인)")

def to_binary_series(y: pd.Series) -> pd.Series:
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        z = y.astype(str).str.strip().str.lower()
        mapping = {"yes":1,"no":0,"y":1,"n":0,"true":1,"false":0,"1":1,"0":0}
        out = z.map(mapping)
        if out.isna().any():  # 매핑 안 된 경우: 고유값 2개면 0/1로 임의 매핑
            uniq = z.dropna().unique().tolist()
            if len(uniq) == 2:
                return z.map({uniq[0]:0, uniq[1]:1}).astype(int)
            else:
                raise ValueError(f"타깃이 이진이 아닙니다: {uniq}")
        return out.astype(int)
    # 숫자형이면 0/1 아니면 0.5 기준 이진화
    vals = pd.unique(y.dropna())
    if set(vals).issubset({0,1}):
        return y.astype(int)
    return (y.astype(float) >= 0.5).astype(int)

def split_cols(df, target_col):
    # object → 범주, 나머지 → 숫자
    feat_cols = [c for c in df.columns if c != target_col]
    num_cols = df[feat_cols].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in feat_cols if c not in num_cols]
    return num_cols, cat_cols

def preprocess_train(df, target_col):
    # id/인덱스 계열 제거
    drop_like = {"id","index","idx","일련번호","번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 타깃
    y = to_binary_series(df[target_col]).values.reshape(-1,1).astype(np.float32)

    # 특징 분리
    num_cols, cat_cols = split_cols(df, target_col)

    # 숫자 결측치 중앙값 대체
    Xn = df[num_cols].copy()
    if len(num_cols) > 0:
        Xn = Xn.apply(pd.to_numeric, errors="coerce")
        Xn = Xn.fillna(Xn.median(numeric_only=True))
    else:
        Xn = pd.DataFrame(index=df.index)

    # 범주 원-핫
    if len(cat_cols) > 0:
        Xc = pd.get_dummies(df[cat_cols].astype(str).fillna("NA"), drop_first=True)
    else:
        Xc = pd.DataFrame(index=df.index)

    # 숫자 표준화
    scaler = None
    if len(num_cols) > 0:
        scaler = StandardScaler()
        Xn_scaled = pd.DataFrame(
            scaler.fit_transform(Xn), columns=num_cols, index=df.index
        ).astype(np.float32)
    else:
        Xn_scaled = Xn

    # 결합 및 최종 배열
    X_all = pd.concat([Xn_scaled, Xc.astype(np.float32)], axis=1)
    feature_names = X_all.columns.tolist()
    return X_all.values.astype(np.float32), y, scaler, feature_names

# ---------- 데이터 로드 & 전처리 ----------
df = load_bank_csv(DATA_DIR)
target = guess_target_column(df)
print(f"[INFO] 타깃 컬럼: {target}")

X_all, y_all, scaler, feat_names = preprocess_train(df, target)

# 학습/검증 분리(계층적)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_SEED
)
n_features = X_tr.shape[1]

# ---------- TF1: ReLU + Dropout MLP (Binary) ----------
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 드롭아웃 제어 (훈련: <1.0, 평가: 1.0)
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
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.sigmoid(logits)  # 예측 확률

# ----- 손실 -----
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yph, logits=logits))

# (선택) 클래스 불균형 시 가중 BCE:
# pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
# pos_weight = neg / max(pos, 1.0)
# loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=yph, logits=logits, pos_weight=pos_weight))

# 옵티마이저
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(loss)

# ----- 지표 -----
pred_bin = tf.cast(probs > 0.5, tf.float32)
acc_tf   = tf.reduce_mean(tf.cast(tf.equal(pred_bin, yph), tf.float32))

# ----- 학습 -----
steps_per_epoch = int(math.ceil(X_tr.shape[0] / BATCH_SIZE))
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, EPOCHS + 1):
        idx = np.random.permutation(X_tr.shape[0])
        Xb, yb = X_tr[idx], y_tr[idx]

        for step in range(steps_per_epoch):
            s = step * BATCH_SIZE
            e = min((step + 1) * BATCH_SIZE, X_tr.shape[0])
            feed = {x: Xb[s:e], yph: yb[s:e], keep_prob1: 0.9, keep_prob2: 0.9}  # ▶ 드롭아웃 ON
            sess.run(train_op, feed_dict=feed)

        if epoch % 50 == 0 or epoch == 1:
            tr_loss, tr_acc = sess.run(
                [loss, acc_tf],
                feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0}  # ▶ 드롭아웃 OFF
            )
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

    # 최종 예측 (드롭아웃 OFF)
    p_train, b_train = sess.run([probs, pred_bin], feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0})
    p_test,  b_test  = sess.run([probs, pred_bin],  feed_dict={x: X_te, yph: y_te, keep_prob1: 1.0, keep_prob2: 1.0})

# ----- Sklearn 지표 -----
acc_tr = accuracy_score(y_tr, b_train)
acc_te = accuracy_score(y_te,  b_test)
auc_te = roc_auc_score(y_te, p_test)
cm_te  = confusion_matrix(y_te, b_test)

print("\n=== Kaggle Bank (ReLU + Dropout MLP) ===")
print(f"Train Acc: {acc_tr:.4f}")
print(f" Test Acc: {acc_te:.4f} | ROC-AUC: {auc_te:.4f}")
print(" Confusion Matrix (test):\n", cm_te)

# === Kaggle Bank (ReLU + Dropout MLP) ===
# Train Acc: 0.9993
#  Test Acc: 0.8307 | ROC-AUC: 0.8159
#  Confusion Matrix (test):
#  [[23865  2158]
#  [ 3429  3555]]
