# TF1-style MLP Regression on sklearn diabetes (ReLU + Dropout)
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 그래프 모드 & 시드
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터
data = load_diabetes()
X = data.data.astype(np.float32)                  # (442, 10)
y = data.target.reshape(-1, 1).astype(np.float32) # 연속 타깃

# 2) 분할 & 스케일링(X만)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7777
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)
n_features = X_train.shape[1]

# 3) 플레이스홀더
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# ✅ 드롭아웃 제어(훈련: <1.0, 평가: 1.0)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

# ReLU에 맞는 He 초기화
he = tf.compat.v1.keras.initializers.he_uniform()

# 4) MLP: 10 -> 128 -> 64 -> 1  (은닉 ReLU + Dropout)
w1 = tf.compat.v1.Variable(he(shape=[n_features, 128])); b1 = tf.compat.v1.Variable(tf.zeros([128]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)   # ▶ 훈련 중 일부 뉴런 무작위 비활성

w2 = tf.compat.v1.Variable(he(shape=[128, 64])); b2 = tf.compat.v1.Variable(tf.zeros([64]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

# 출력층: 회귀 → 선형(활성화 X)
w3 = tf.compat.v1.Variable(he(shape=[64, 1])); b3 = tf.compat.v1.Variable(tf.zeros([1]))
pred = tf.matmul(h2, w3) + b3

# (옵션) 예측을 항상 양수로 제한하고 싶다면 다음 한 줄로 교체:
# pred = tf.nn.softplus(tf.matmul(h2, w3) + b3)

# 5) 손실/옵티마이저
mse = tf.reduce_mean(tf.square(pred - yph))
# (선택) L2 정규화 살짝 추가하고 싶다면:
# l2 = 1e-4*(tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3))
# loss = mse + l2
loss = mse
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 6) 학습(미니배치 + 드롭아웃 ON)
epochs = 500
batch_size = 64
steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        # 셔플
        idx = np.random.permutation(X_train.shape[0])
        X_tr, y_tr = X_train[idx], y_train[idx]

        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_train.shape[0])
            xb, yb = X_tr[s:e], y_tr[s:e]
            # ▶ 훈련 시: keep_prob < 1.0 (예: 0.9 = 10% 드롭)
            sess.run(train_op, feed_dict={x: xb, yph: yb, keep_prob1: 0.9, keep_prob2: 0.9})

        if epoch % 50 == 0 or epoch == 1:
            tr_mse = sess.run(mse, feed_dict={x: X_train, yph: y_train, keep_prob1: 1.0, keep_prob2: 1.0})
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {tr_mse:.4f}")

    # 7) 평가/예측(드롭아웃 OFF = 1.0)
    y_tr_pred = sess.run(pred, feed_dict={x: X_train, yph: y_train, keep_prob1: 1.0, keep_prob2: 1.0})
    y_te_pred = sess.run(pred, feed_dict={x: X_test,  yph: y_test,  keep_prob1: 1.0, keep_prob2: 1.0})

# 8) 스코어
from math import sqrt
tr_mse = mean_squared_error(y_train, y_tr_pred); te_mse = mean_squared_error(y_test,  y_te_pred)
tr_mae = mean_absolute_error(y_train, y_tr_pred); te_mae = mean_absolute_error(y_test,  y_te_pred)
tr_rmse, te_rmse = sqrt(tr_mse), sqrt(te_mse)
tr_r2 = r2_score(y_train, y_tr_pred); te_r2 = r2_score(y_test,  y_te_pred)

print("\n=== Diabetes Regression (ReLU + Dropout) ===")
print(f"Train RMSE: {tr_rmse:.3f} | MAE: {tr_mae:.3f} | R²: {tr_r2:.3f}")
print(f" Test RMSE: {te_rmse:.3f} | MAE: {te_mae:.3f} | R²: {te_r2:.3f}")

# === Diabetes Regression (ReLU + Dropout) ===
# Train RMSE: 37.410 | MAE: 28.387 | R²: 0.759
#  Test RMSE: 57.836 | MAE: 44.672 | R²: 0.480
