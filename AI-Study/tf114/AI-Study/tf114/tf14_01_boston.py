import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# tf.random.set_seed(7777)  # 또는 
tf.compat.v1.set_random_seed(7777)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# 스케일/형 맞추기 (y는 (N,1))
x_train = x_train.astype(np.float32)
x_test  = x_test.astype(np.float32)
y_train = y_train.astype(np.float32).reshape(-1, 1)
y_test  = y_test.astype(np.float32).reshape(-1, 1)

# 2. 모델 (TF1 스타일)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([13, 1], stddev=0.1), name='weights')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.matmul(x, w) + b  # (N,1)

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))  # MSE
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)  # 0.1은 너무 큼
train = optimizer.minimize(loss)

# 3-2. 훈련
loss_history = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1000

    for step in range(epochs + 1):
        _, l = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
        loss_history.append(l)
        if step % 100 == 0:
            print(f"{step:4d}  loss={l:.4f}")

    # 최종 파라미터
    w_val, b_val = sess.run([w, b])
    print("="*15, "최종 가중치", "="*15)
    print("w:", w_val.ravel(), "\nb:", b_val)

    # 예측 (훈련/테스트)
    y_train_pred = sess.run(hypothesis, feed_dict={x: x_train})  # (N,1)
    y_test_pred  = sess.run(hypothesis, feed_dict={x: x_test})   # (N,1)

# 4. 메트릭 (1D로 압축해 계산)
ytr, ytrp = y_train.squeeze(), y_train_pred.squeeze()
yte, ytep = y_test.squeeze(),  y_test_pred.squeeze()

train_r2  = r2_score(ytr, ytrp)
train_mae = mean_absolute_error(ytr, ytrp)
test_r2   = r2_score(yte, ytep)
test_mae  = mean_absolute_error(yte, ytep)
test_rmse = np.sqrt(mean_squared_error(yte, ytep))

print(f"\n[Train] R2: {train_r2:.4f} | MAE: {train_mae:.4f}")
print(f"[Test ] R2: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")

# (선택) loss 그래프가 필요하면:
# import matplotlib.pyplot as plt
# plt.plot(loss_history)
# plt.xlabel('epochs'); plt.ylabel('loss (MSE)'); plt.title('Training Loss')
# plt.show()


# =============== 최종 가중치 ===============
# w: [-0.08930459  0.10224833 -0.04907196  4.657574    0.7936011   1.9245182
#   0.07649332 -0.3145661   0.13306358 -0.00500396  0.34626526  0.02121846
#  -0.72653157]
# b: [1.1001714]

# [Train] R2: 0.6091 | MAE: 4.2692
# [Test ] R2: 0.5533 | MAE: 4.4018 | RMSE: 6.0978