import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(9999)

# 1) 데이터 로드
X, y = load_diabetes(return_X_y=True)   # X: (442, 10), y: (442,)
X = X.astype(np.float32)
y = y.astype(np.float32).reshape(-1, 1)  # (N,1)

# 2) train / test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9999
)

# 3) 스케일링(특징만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]

# ----- TF1 그래프 구성 -----
x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1], stddev=0.1), name='weights')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.matmul(x_ph, w) + b                # (N,1)
loss = tf.reduce_mean(tf.square(hypothesis - y_ph))  # MSE

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# 로깅용
loss_history = []
w_norm_history = []

# ----- 학습 -----
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

    # 예측
    y_train_pred = sess.run(hypothesis, feed_dict={x_ph: X_train})
    y_test_pred  = sess.run(hypothesis, feed_dict={x_ph: X_test})

# ----- 메트릭 -----
ytr, ytrp = y_train.squeeze(), y_train_pred.squeeze()
yte, ytep = y_test.squeeze(),  y_test_pred.squeeze()

train_r2  = r2_score(ytr, ytrp)
train_mae = mean_absolute_error(ytr, ytrp)
test_r2   = r2_score(yte, ytep)
test_mae  = mean_absolute_error(yte, ytep)
test_rmse = np.sqrt(mean_squared_error(yte, ytep))

print(f"\n[Train] R2: {train_r2:.4f} | MAE: {train_mae:.4f}")
print(f"[Test ] R2: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")

# ----- 3개 subplot -----
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


# [Train] R2: -2.6047 | MAE: 132.0093
# [Test ] R2: -2.3374 | MAE: 141.6151 | RMSE: 153.6686