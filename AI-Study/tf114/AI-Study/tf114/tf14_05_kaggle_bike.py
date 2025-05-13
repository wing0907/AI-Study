import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------- 설정 ---------
DATA_DIR = r"D:\Study25WJ\_data\kaggle\bike"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")  # 아래 예측/저장은 전부 주석 처리
TARGET_COL = "count"

# TF1 스타일
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(9999)

# --------- 유틸: CSV 읽기(한글/인코딩 대비) ---------
def read_csv_kr(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

# --------- 데이터 로드 ---------
assert os.path.exists(TRAIN_CSV), f"train.csv가 경로에 없습니다: {TRAIN_CSV}"
train = read_csv_kr(TRAIN_CSV)

# --------- 전처리 ---------
df = train.copy()

# datetime 파생
if "datetime" in df.columns:
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df.drop(columns=["datetime"], inplace=True)

# Kaggle bike는 train에 'casual','registered'가 있으며, 타깃 'count'와 누설 관계 → 제거
for leak_col in ["casual", "registered"]:
    if leak_col in df.columns:
        df.drop(columns=[leak_col], inplace=True)

# 타깃/피처 분리
assert TARGET_COL in df.columns, f"'{TARGET_COL}' 컬럼을 찾지 못했습니다. (columns={list(df.columns)})"
y = df[TARGET_COL].astype(np.float32).values            # (N,) 1D
X = df.drop(columns=[TARGET_COL])

# 숫자 피처만 사용
X = X.select_dtypes(include=[np.number]).astype(np.float32)

# 결측치 처리(있다면)
X = X.fillna(X.median())

# 학습/검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# 스케일링(특징만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_valid = scaler.transform(X_valid).astype(np.float32)

n_features = X_train.shape[1]

# --------- TF1 그래프(모두 1D) ---------
x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])  # (N, D)
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])              # (N,)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features], stddev=0.1), name="weights")  # (D,)
b = tf.compat.v1.Variable(0.0, name="bias")  # 스칼라

# 예측: (N,D) · (D,) + () → (N,)
hypothesis = tf.tensordot(x_ph, w, axes=1) + b

loss = tf.reduce_mean(tf.square(hypothesis - y_ph))  # MSE
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# 로깅
loss_history = []
w_norm_history = []

# --------- 학습 ---------
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    EPOCHS = 2000
    for step in range(EPOCHS + 1):
        _, l, w_val, b_val, w_norm = sess.run(
            [train_op, loss, w, b, tf.norm(w)],
            feed_dict={x_ph: X_train, y_ph: y_train}
        )
        loss_history.append(l)
        w_norm_history.append(w_norm)

        if step % 200 == 0:
            print(f"{step:4d}  loss={l:.4f}  ||w||={w_norm:.4f}")

    # 예측 (1D)
    y_tr_pred = sess.run(hypothesis, feed_dict={x_ph: X_train})  # (N,)
    y_va_pred = sess.run(hypothesis, feed_dict={x_ph: X_valid})  # (N,)

# --------- 메트릭 ---------
train_r2  = r2_score(y_train, y_tr_pred)
train_mae = mean_absolute_error(y_train, y_tr_pred)
valid_r2  = r2_score(y_valid, y_va_pred)
valid_mae = mean_absolute_error(y_valid, y_va_pred)
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_va_pred))

print(f"\n[Train] R2: {train_r2:.4f} | MAE: {train_mae:.4f}")
print(f"[Valid] R2: {valid_r2:.4f} | MAE: {valid_mae:.4f} | RMSE: {valid_rmse:.4f}")

# --------- 3개 subplot ---------
plt.figure(figsize=(15, 4.5))

# 1) Loss vs Epoch
plt.subplot(1, 3, 1)
plt.plot(loss_history)
plt.xlabel('epochs'); plt.ylabel('loss (MSE)')
plt.title('Loss vs Epoch')

# 2) ||W|| vs Epoch
plt.subplot(1, 3, 2)
plt.plot(w_norm_history)
plt.xlabel('epochs'); plt.ylabel('||weights||_2')
plt.title('Weight Norm vs Epoch')
plt.grid(True)

# 3) Loss vs ||W||
plt.subplot(1, 3, 3)
plt.plot(w_norm_history, loss_history)
plt.xlabel('||weights||_2'); plt.ylabel('loss (MSE)')
plt.title('Loss vs Weight Norm')
plt.grid(True)

plt.tight_layout()
plt.show()

# --------- (선택) test.csv 예측/저장 — 요청 시 활성화 (현재 전부 주석 처리) ---------
# if os.path.exists(TEST_CSV):
#     test = read_csv_kr(TEST_CSV)
#     if "datetime" in test.columns:
#         dt = pd.to_datetime(test["datetime"], errors="coerce")
#         test["year"] = dt.dt.year
#         test["month"] = dt.dt.month
#         test["day"] = dt.dt.day
#         test["hour"] = dt.dt.hour
#         test["dayofweek"] = dt.dt.dayofweek
#         test.drop(columns=["datetime"], inplace=True)
#     for leak_col in ["casual", "registered"]:
#         if leak_col in test.columns:
#             test.drop(columns=[leak_col], inplace=True)
#     test_X = test.select_dtypes(include=[np.number]).astype(np.float32)
#     test_X = test_X.fillna(test_X.median())
#     test_X = scaler.transform(test_X).astype(np.float32)
#     y_test_pred = test_X @ w_val + b_val
#     sub = pd.DataFrame({"count": y_test_pred})
#     if "id" in test.columns:
#         sub.insert(0, "id", test["id"].values)
#     out_path = os.path.join(DATA_DIR, "submission_linear_tf.csv")
#     sub.to_csv(out_path, index=False, encoding="utf-8-sig")
#     print(f"\nsubmission 저장 완료: {out_path}")


# [Train] R2: -0.6153 | MAE: 174.0907
# [Valid] R2: -0.6137 | MAE: 175.1429 | RMSE: 230.7851