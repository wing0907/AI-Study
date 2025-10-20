# TF1-style Softmax Regression on fetch_covetype (7 classes)
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import math

# --- TF1 그래프 모드 설정 & 시드 ---
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드 (X: (581012, 54), y: 1..7)
cov = fetch_covtype()
X_all = cov.data.astype(np.float32)
y_all = cov.target.astype(np.int64)

# y를 0..6으로 변환, 원-핫 인코딩
num_classes = 7
y_int = y_all - 1
y_oh = np.eye(num_classes, dtype=np.float32)[y_int]

# 2) 학습/검증 분할 (계층적 분할)
X_train, X_test, y_train_oh, y_test_oh, y_train_int, y_test_int = train_test_split(
    X_all, y_oh, y_int, test_size=0.2, stratify=y_int, random_state=7777
)

# 3) 스케일링 (수렴 안정)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 54

# 4) TF1 그래프 정의
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])

w = tf.compat.v1.Variable(tf.random.normal([n_features, num_classes]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.zeros([num_classes]), name='bias', dtype=tf.float32)

logits = tf.matmul(x, w) + b
probs  = tf.nn.softmax(logits)

# 안정적인 softmax CE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))

# 대용량이므로 학습 안정성을 위해 Adam 사용 (원하면 GD로 바꿔도 됨)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
train_op  = optimizer.minimize(loss)

# 정확도 (argmax 비교)
pred_idx = tf.argmax(probs, axis=1)
true_idx = tf.argmax(y_ph, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# 5) 미니배치 학습 루프
epochs = 25
batch_size = 4096
steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        # 셔플
        idx = np.random.permutation(X_train.shape[0])
        X_train_shuf = X_train[idx]
        y_train_shuf = y_train_oh[idx]

        epoch_loss = 0.0
        epoch_acc  = 0.0

        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_train.shape[0])
            xb = X_train_shuf[s:e]
            yb = y_train_shuf[s:e]

            _, lval, aval = sess.run([train_op, loss, acc], feed_dict={x: xb, y_ph: yb})
            epoch_loss += lval * (e - s)
            epoch_acc  += aval * (e - s)

        epoch_loss /= X_train.shape[0]
        epoch_acc  /= X_train.shape[0]

        if epoch % 1 == 0:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc(TF): {epoch_acc:.4f}")

    # 6) 최종 성능 측정
    tr_acc_tf = sess.run(acc, feed_dict={x: X_train, y_ph: y_train_oh})
    te_acc_tf = sess.run(acc, feed_dict={x: X_test,  y_ph: y_test_oh})

    # sklearn accuracy (참고용)
    y_test_pred_idx = sess.run(pred_idx, feed_dict={x: X_test})
    sk_acc_test = accuracy_score(y_test_int, y_test_pred_idx)

    print("\n=== Results ===")
    print(f"Train Acc (TF): {tr_acc_tf:.4f}")
    print(f" Test Acc (TF): {te_acc_tf:.4f}")
    print(f" Test Acc (sklearn): {sk_acc_test:.4f}")

# === Results ===
# Train Acc (TF): 0.7215
#  Test Acc (TF): 0.7208
#  Test Acc (sklearn): 0.7208