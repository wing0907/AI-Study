import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------- 설정 ---------
DATA_DIR = r"D:\Study25WJ\_data\dacon\따릉이"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")  # 있으면 예측/저장 (아래서 주석처리)
TARGET_COL = "count"  # 보통 Dacon 따릉이는 'count'가 정답컬럼

# TF1 스타일을 TF2에서 쓰려면 필요
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(9999)

# --------- 유틸: CSV 읽기(한글 경로/인코딩 대비) ---------
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

# datetime 파생 (있을 경우)
if "datetime" in df.columns:
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    # 원본 datetime은 제거
    df.drop(columns=["datetime"], inplace=True)

# Yes/No 유형 처리 (holiday/functioning_day 등)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].map({"Yes": 1, "No": 0}).astype("float32")

# 타깃/피처 분리
assert TARGET_COL in df.columns, f"'{TARGET_COL}' 컬럼을 찾지 못했습니다. 실제 타깃 컬럼명을 TARGET_COL에 지정해주세요. (columns={list(df.columns)})"
y = df[TARGET_COL].astype(np.float32).values          # ← (N,) 1D
X = df.drop(columns=[TARGET_COL])

# 숫자 이외 제거(남아있다면)
X = X.select_dtypes(include=[np.number]).astype(np.float32)

# 결측치 처리
X = X.fillna(X.median())

# 학습/검증 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# 표준화(특징만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_valid = scaler.transform(X_valid).astype(np.float32)

n_features = X_train.shape[1]

# --------- TF1 그래프 구성 (모두 1D 형태) ---------
x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])  # (N, D)
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])              # (N,) 1D

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features], stddev=0.1), name="weights")  # (D,) 1D
b = tf.compat.v1.Variable(0.0, name="bias")  # 스칼라

# (N,D) · (D,) + ()  ->  (N,) 1D
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

    EPOCHS = 1500
    for step in range(EPOCHS + 1):
        _, l, w_val, b_val, w_norm = sess.run(
            [train_op, loss, w, b, tf.norm(w)],
            feed_dict={x_ph: X_train, y_ph: y_train}
        )
        loss_history.append(l)
        w_norm_history.append(w_norm)

        if step % 100 == 0:
            print(f"{step:4d}  loss={l:.4f}  ||w||={w_norm:.4f}")

    # 예측 (1D로 반환)
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

# --------- (선택) test.csv 있으면 예측/저장 — 요청대로 전체 주석처리 ---------
# if os.path.exists(TEST_CSV):
#     test = read_csv_kr(TEST_CSV)
# 
#     if "datetime" in test.columns:
#         dt = pd.to_datetime(test["datetime"], errors="coerce")
#         test["year"] = dt.dt.year
#         test["month"] = dt.dt.month
#         test["day"] = dt.dt.day
#         test["hour"] = dt.dt.hour
#         test["dayofweek"] = dt.dt.dayofweek
#         test.drop(columns=["datetime"], inplace=True)
# 
#     for col in test.columns:
#         if test[col].dtype == "object":
#             test[col] = test[col].map({"Yes": 1, "No": 0}).astype("float32")
# 
#     test_X = test.select_dtypes(include=[np.number]).astype(np.float32)
#     test_X = test_X.fillna(test_X.median())
#     test_X = scaler.transform(test_X).astype(np.float32)
# 
#     # 학습된 w_val, b_val 사용 (둘 다 1D/스칼라)
#     y_test_pred = test_X @ w_val + b_val     # (N,) 1D
# 
#     sub = pd.DataFrame({"count": y_test_pred})
#     if "id" in test.columns:
#         sub.insert(0, "id", test["id"].values)
# 
#     out_path = os.path.join(DATA_DIR, "submission_linear_tf.csv")
#     sub.to_csv(out_path, index=False, encoding="utf-8-sig")
#     print(f"\nsubmission 저장 완료: {out_path}")
