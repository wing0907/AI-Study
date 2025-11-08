# TF1-style MLP Regression on California Housing
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 그래프 모드 & 시드 고정
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
data = fetch_california_housing()
X = data.data.astype(np.float32)          # (20640, 8)
y = data.target.reshape(-1, 1).astype(np.float32)  # 중위 주택가(단위: $100k)

# 2) 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7777
)

# 3) 입력 표준화(권장) — y는 원 스케일 유지
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 8

# 4) 플레이스홀더
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 5) MLP 구성: 8 -> 128 -> 64 -> 1 (출력층은 선형)
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

w1 = tf.compat.v1.Variable(glorot(shape=[n_features, 128]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')
h1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(glorot(shape=[128, 64]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')
h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)

w3 = tf.compat.v1.Variable(glorot(shape=[64, 1]), name='w3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='b3')
pred = tf.matmul(h2, w3) + b3  # 회귀이므로 활성화 없음(= 선형 출력)

# 6) 손실/옵티마이저
mse = tf.reduce_mean(tf.square(pred - y_ph))  # MSE
# (선택) L2 정규화: 아주 살짝 가중치에 패널티를 주고 싶다면 주석 해제
# l2 = 1e-4 * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
# loss = mse + l2
loss = mse

train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 7) 미니배치 학습
epochs = 300
batch_size = 512
steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        # 셔플
        idx = np.random.permutation(X_train.shape[0])
        X_tr = X_train[idx]
        y_tr = y_train[idx]

        # 한 에폭 학습
        epoch_mse = 0.0
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_train.shape[0])
            xb, yb = X_tr[s:e], y_tr[s:e]
            _, mse_val = sess.run([train_op, mse], feed_dict={x: xb, y_ph: yb})
            epoch_mse += mse_val * (e - s)
        epoch_mse /= X_train.shape[0]

        if epoch % 20 == 0 or epoch == 1:
            tr_mse = sess.run(mse, feed_dict={x: X_train, y_ph: y_train})
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {tr_mse:.4f}")

    # 8) 최종 예측
    y_tr_pred = sess.run(pred, feed_dict={x: X_train})
    y_te_pred = sess.run(pred, feed_dict={x: X_test})

# 9) 스코어 계산
tr_mse = mean_squared_error(y_train, y_tr_pred)
te_mse = mean_squared_error(y_test,  y_te_pred)
tr_mae = mean_absolute_error(y_train, y_tr_pred)
te_mae = mean_absolute_error(y_test,  y_te_pred)
tr_rmse = np.sqrt(tr_mse)
te_rmse = np.sqrt(te_mse)
tr_r2 = r2_score(y_train, y_tr_pred)
te_r2 = r2_score(y_test,  y_te_pred)

print("\n=== Results (California Housing) ===")
print(f"Train RMSE: {tr_rmse:.3f} | MAE: {tr_mae:.3f} | R²: {tr_r2:.3f}")
print(f" Test RMSE: {te_rmse:.3f} | MAE: {te_mae:.3f} | R²: {te_r2:.3f}")

# === Results (California Housing) ===
# Train RMSE: 0.494 | MAE: 0.338 | R²: 0.816
#  Test RMSE: 0.502 | MAE: 0.343 | R²: 0.812