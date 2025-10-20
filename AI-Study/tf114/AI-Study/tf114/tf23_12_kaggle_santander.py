# -*- coding: utf-8 -*-
import os, warnings, math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ========== 설정 ==========
DATA_DIR   = r"D:\Study25WJ\_data\kaggle\santander"
RANDOM_SEED= 7777
EPOCHS     = 40          # 데이터가 크면 20~60 사이로 조절
BATCH_SIZE = 4096        # 여유되면 8192
LR         = 1e-3

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

def load_train_csv(data_dir):
    """
    우선순위:
      1) train.csv
      2) santander_train.csv
      3) 폴더 내 유일 CSV
    """
    for name in ["train.csv", "santander_train.csv"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            print(f"[INFO] 사용 파일: {p}")
            return read_csv_kr(p)
    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if len(csvs) == 1:
        p = os.path.join(data_dir, csvs[0])
        print(f"[INFO] 사용 파일(자동): {p}")
        return read_csv_kr(p)
    raise FileNotFoundError("학습용 CSV를 찾지 못했습니다. (train.csv 권장)")

# ---------- 전처리 ----------
def guess_target_column(df):
    for c in ["target","TARGET","Target","y","Y","label","Label","Class","class"]:
        if c in df.columns:
            return c
    # 마지막에 고유값이 2개인 열을 뒤에서부터 탐색
    for c in df.columns[::-1]:
        if df[c].nunique(dropna=True) == 2:
            return c
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (target / y / label 등 확인)")

def to_binary_series(y: pd.Series) -> pd.Series:
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        z = y.astype(str).strip().lower()
        mapping = {"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0}
        out = y.astype(str).str.strip().str.lower().map(mapping)
        if out.isna().any():
            uniq = y.astype(str).str.strip().str.lower().dropna().unique().tolist()
            if len(uniq) == 2:
                return y.astype(str).str.strip().str.lower().map({uniq[0]:0, uniq[1]:1}).astype(int)
            else:
                raise ValueError(f"타깃이 이진이 아닙니다: {uniq}")
        return out.astype(int)
    # 숫자형이면 0/1 그대로, 아니면 0.5 기준 이진화
    vals = pd.unique(y.dropna())
    if set(vals).issubset({0,1}):
        return y.astype(int)
    return (y.astype(float) >= 0.5).astype(int)

def preprocess(df, target_col):
    # ID/인덱스 계열 제거
    drop_like = {"id","id_code","idcode","index","idx","일련번호","번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 타깃 가공
    y = to_binary_series(df[target_col]).values.reshape(-1,1).astype(np.float32)

    # 특징 칼럼(숫자 위주)
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].copy()

    # 전부 숫자화 시도 (Santander는 보통 전부 float)
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # 결측은 중앙값으로
    X = X.fillna(X.median(numeric_only=True))

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, y, scaler, feat_cols

# ---------- 데이터 로드 & 분리 ----------
df = load_train_csv(DATA_DIR)
target = guess_target_column(df)
print(f"[INFO] 타깃 컬럼: {target}")

X_all, y_all, scaler, feat_names = preprocess(df, target)

# 데이터 크기가 크므로 stratify로 분할
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

# ReLU + He 초기화
he = tf.compat.v1.keras.initializers.he_uniform()

# n_features -> 512 -> 256 -> 1  (대용량 기준 넉넉하게)
w1 = tf.compat.v1.Variable(he(shape=[n_features, 512])); b1 = tf.compat.v1.Variable(tf.zeros([512]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)

w2 = tf.compat.v1.Variable(he(shape=[512, 256])); b2 = tf.compat.v1.Variable(tf.zeros([256]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(he(shape=[256, 1])); b3 = tf.compat.v1.Variable(tf.zeros([1]))
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.sigmoid(logits)  # 예측 확률

# ----- 손실 -----
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yph, logits=logits))

# (선택) 클래스 불균형 시 가중 BCE 사용:
# pos = float((y_tr == 1).sum()); neg = float((y_tr == 0).sum())
# pos_weight = neg / max(pos, 1.0)
# loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=yph, logits=logits, pos_weight=pos_weight))

# 옵티마이저
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(loss)

# ----- 지표 -----
pred_bin = tf.cast(probs > 0.5, tf.float32)
acc_tf   = tf.reduce_mean(tf.cast(tf.equal(pred_bin, yph), tf.float32))

# ----- 학습 루프 (미니배치 + 드롭아웃 ON) -----
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

        if epoch % 5 == 0 or epoch == 1:
            tr_loss, tr_acc = sess.run(
                [loss, acc_tf],
                feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0}  # ▶ 평가: 드롭아웃 OFF
            )
            print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

    # 최종 예측 (드롭아웃 OFF)
    p_train, b_train = sess.run([probs, pred_bin], feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0})
    p_test,  b_test  = sess.run([probs, pred_bin],  feed_dict={x: X_te, yph: y_te, keep_prob1: 1.0, keep_prob2: 1.0})

# ----- Sklearn 지표 -----
acc_tr = accuracy_score(y_tr, b_train)
acc_te = accuracy_score(y_te, b_test)
auc_te = roc_auc_score(y_te, p_test)
cm_te  = confusion_matrix(y_te, b_test)

print("\n=== Santander (ReLU + Dropout MLP) ===")
print(f"Train Acc: {acc_tr:.4f}")
print(f" Test  Acc: {acc_te:.4f} | ROC-AUC: {auc_te:.4f}")
print(" Confusion Matrix (test):\n", cm_te)

# === Santander (ReLU + Dropout MLP) ===
# Train Acc: 0.9920
#  Test  Acc: 0.9028 | ROC-AUC: 0.7998
#  Confusion Matrix (test):
#  [[34990   990]
#  [ 2899  1121]]