# TF1-style MLP on Kaggle Otto (ReLU + Dropout, 9 classes)
# -*- coding: utf-8 -*-
import os, warnings, math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ===== 설정 =====
DATA_DIR   = r"D:\Study25WJ\_data\kaggle\otto"
RANDOM_SEED= 7777
EPOCHS     = 40           # 데이터 크기에 맞춰 30~60 사이에서 조정 권장
BATCH_SIZE = 1024         # 여유되면 2048~4096
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
    """우선순위: train.csv -> otto_train.csv -> 폴더 내 유일 CSV"""
    for name in ["train.csv", "otto_train.csv"]:
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
    for c in ["target","TARGET","Target","label","Label","Class","class","y","Y"]:
        if c in df.columns:
            return c
    # 마지막에 문자열/범주형인 열 탐색
    for c in df.columns[::-1]:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (target/label/… 확인)")

def preprocess_train(df, target_col):
    # ID 계열 제거
    drop_like = {"id","ID","Id","id_code","index","idx","일련번호","번호"}
    drop_cols = [c for c in df.columns if c in drop_like or c.lower() in drop_like]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 타깃 인코딩: 'Class_1'~'Class_9' → 0..8
    y_str = df[target_col].astype(str)
    classes = sorted(y_str.unique())
    cls2idx = {c:i for i,c in enumerate(classes)}
    y_int = y_str.map(cls2idx).astype(np.int64).values
    num_classes = len(classes)
    y_oh = np.eye(num_classes, dtype=np.float32)[y_int]

    # 특징(숫자화 + 결측 중앙값 + 표준화)
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].copy()
    for c in feat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, y_oh, y_int, scaler, feat_cols, classes

# ---------- 데이터 로드 & 분리 ----------
df = load_train_csv(DATA_DIR)
target = guess_target_column(df)
print(f"[INFO] 타깃 컬럼: {target}")

X_all, y_all, y_int_all, scaler, feat_names, classes = preprocess_train(df, target)
print("[INFO] 클래스 매핑:", {i:c for i,c in enumerate(classes)})

# 학습/검증 분리(계층적)
X_tr, X_te, y_tr, y_te, yi_tr, yi_te = train_test_split(
    X_all, y_all, y_int_all, test_size=0.2, stratify=y_int_all, random_state=RANDOM_SEED
)
n_features  = X_tr.shape[1]
num_classes = y_tr.shape[1]

# ---------- TF1: ReLU + Dropout MLP (Multiclass) ----------
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])

# 드롭아웃(훈련: <1.0, 평가: 1.0)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

# He 초기화 (ReLU)
he = tf.compat.v1.keras.initializers.he_uniform()

# 구조: n_features -> 512 -> 256 -> num_classes
w1 = tf.compat.v1.Variable(he(shape=[n_features, 512])); b1 = tf.compat.v1.Variable(tf.zeros([512]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)

w2 = tf.compat.v1.Variable(he(shape=[512, 256])); b2 = tf.compat.v1.Variable(tf.zeros([256]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(he(shape=[256, num_classes])); b3 = tf.compat.v1.Variable(tf.zeros([num_classes]))
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.softmax(logits)

# 손실/옵티마이저
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yph, logits=logits))
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(loss)

# 정확도
pred_idx = tf.argmax(probs, axis=1)
true_idx = tf.argmax(yph, axis=1)
acc_tf   = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# ---------- 학습 루프 ----------
epochs = EPOCHS
batch_size = BATCH_SIZE
steps_per_epoch = math.ceil(X_tr.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(X_tr.shape[0])
        Xb, yb = X_tr[idx], y_tr[idx]

        epoch_loss = 0.0
        epoch_acc  = 0.0

        # ▶ 훈련: 드롭아웃 ON (예: keep_prob=0.9 → 10% 드롭)
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_tr.shape[0])
            feed = {x: Xb[s:e], yph: yb[s:e], keep_prob1: 0.9, keep_prob2: 0.9}
            _, lval, aval = sess.run([train_op, loss, acc_tf], feed_dict=feed)
            epoch_loss += lval * (e - s)
            epoch_acc  += aval * (e - s)

        epoch_loss /= X_tr.shape[0]
        epoch_acc  /= X_tr.shape[0]

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # 평가(드롭아웃 OFF)
    tr_acc = sess.run(acc_tf, feed_dict={x: X_tr, yph: y_tr, keep_prob1: 1.0, keep_prob2: 1.0})
    te_pred = sess.run(pred_idx, feed_dict={x: X_te, yph: y_te, keep_prob1: 1.0, keep_prob2: 1.0})

# ---------- Sklearn 지표 ----------
acc_te = accuracy_score(yi_te, te_pred)
cm_te  = confusion_matrix(yi_te, te_pred)

print("\n=== Otto (ReLU + Dropout MLP) ===")
print(f"Train Acc (TF): {tr_acc:.4f}")
print(f" Test Acc: {acc_te:.4f}")
print(" Confusion Matrix (test):\n", cm_te)
print("Label mapping (index -> original):", {i:c for i,c in enumerate(classes)})

# ---------- (선택) 제출 저장 예시 — 요청 없어서 미생성 ----------
# test.csv를 불러서 같은 스키마로 전처리하고, probs/pred로 submission 만들 수 있습니다.
# 데이터셋 경로/칼럼이 확정되면 주석을 풀고 사용하세요.
# test_path = os.path.join(DATA_DIR, "test.csv")
# if os.path.exists(test_path):
#     df_test = read_csv_kr(test_path)
#     # ID 보존
#     test_id = df_test["id"] if "id" in df_test.columns else np.arange(len(df_test))
#     # 학습 스키마에 맞춘 전처리(스케일러/피처 정렬 동일 적용)가 필요합니다.
#     # (간단화를 위해 여기선 생략)
#     pass


# === Otto (ReLU + Dropout MLP) ===
# Train Acc (TF): 0.9115
#  Test Acc: 0.8132
#  Confusion Matrix (test):
#  [[ 218    4    6    3    0   24   23   44   64]
#  [   4 2558  537   77    4    6   25    5    8]
#  [   2  560  943   56    1    2   33    3    1]
#  [   0  108   82  316    3   13   14    0    2]
#  [   3    4    0    1  535    0    5    0    0]
#  [  22   16    2    5    1 2665   35   54   27]
#  [  17   34   49    9    0   25  407   24    3]
#  [  38   16    2    0    2   30   23 1559   23]
#  [  47   11    1    0    2   26   10   31  863]]
# Label mapping (index -> original): {0: 'Class_1', 1: 'Class_2', 2: 'Class_3', 3: 'Class_4', 4: 'Class_5', 5: 'Class_6', 6: 'Class_7', 7: 'Class_8', 8: 'Class_9'}