# TF1-style MLP Regression on Boston Housing
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# 그래프 모드 & 시드
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드 (train/test 이미 분리되어 제공)
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test  = y_test.astype(np.float32).reshape(-1, 1)

# 2) 입력 표준화 (y는 원 스케일 유지)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 13

# 3) 플레이스홀더
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 4) MLP 구성: 13 -> 64 -> 32 -> 1
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

w1 = tf.compat.v1.Variable(glorot(shape=[n_features, 64]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([64]), name='b1')
h1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(glorot(shape=[64, 32]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([32]), name='b2')
h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)

# 출력층은 회귀이므로 활성화 없음(= 선형)
w3 = tf.compat.v1.Variable(glorot(shape=[32, 1]), name='w3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='b3')
pred = tf.matmul(h2, w3) + b3  # 예측값 (주택가격, 단위: $1000s)

# 5) 손실/옵티마이저
mse = tf.reduce_mean(tf.square(pred - y))  # MSE
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(mse)

# (참고용) MAE 텐서도 만들어 두기
mae = tf.reduce_mean(tf.abs(pred - y))

# 6) 미니배치 학습 루프
epochs = 500
batch_size = 32
steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        # 셔플
        idx = np.random.permutation(X_train.shape[0])
        X_tr = X_train[idx]
        y_tr = y_train[idx]

        epoch_mse = 0.0
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_train.shape[0])
            xb, yb = X_tr[s:e], y_tr[s:e]
            _, mse_val = sess.run([train_op, mse], feed_dict={x: xb, y: yb})
            epoch_mse += mse_val * (e - s)
        epoch_mse /= X_train.shape[0]

        if epoch % 50 == 0 or epoch == 1:
            tr_mse, tr_mae = sess.run([mse, mae], feed_dict={x: X_train, y: y_train})
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE: {tr_mse:.4f} | MAE: {tr_mae:.4f}")

    # 7) 최종 평가 (넘파이로 변환해 지표 계산)
    y_tr_pred = sess.run(pred, feed_dict={x: X_train})
    y_te_pred = sess.run(pred, feed_dict={x: X_test})

# 8) 스코어 계산 (원 스케일)
tr_mse = mean_squared_error(y_train, y_tr_pred)
te_mse = mean_squared_error(y_test,  y_te_pred)
tr_mae = mean_absolute_error(y_train, y_tr_pred)
te_mae = mean_absolute_error(y_test,  y_te_pred)
tr_rmse = np.sqrt(tr_mse)
te_rmse = np.sqrt(te_mse)
tr_r2 = r2_score(y_train, y_tr_pred)
te_r2 = r2_score(y_test,  y_te_pred)

print("\n=== Results (Boston Housing) ===")
print(f"Train RMSE: {tr_rmse:.3f} | MAE: {tr_mae:.3f} | R²: {tr_r2:.3f}")
print(f" Test RMSE: {te_rmse:.3f} | MAE: {te_mae:.3f} | R²: {te_r2:.3f}")


# === Results (Boston Housing) ===
# Train RMSE: 1.202 | MAE: 0.876 | R²: 0.983
#  Test RMSE: 3.824 | MAE: 2.219 | R²: 0.824
