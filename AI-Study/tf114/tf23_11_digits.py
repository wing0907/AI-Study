# TF1-style MLP on sklearn digits (ReLU + Dropout, 10 classes)
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import math

# --- TF1 그래프 모드 & 시드 ---
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
digits = load_digits()
X = digits.data.astype(np.float32)     # (1797, 64)  각 픽셀 0~16
y_int = digits.target                  # (1797,)     0..9
num_classes = 10
y = np.eye(num_classes, dtype=np.float32)[y_int]  # (1797, 10) 원-핫

# 2) 학습/검증 분리(계층적)
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y, y_int, test_size=0.2, stratify=y_int, random_state=7777
)

# 3) 표준화(입력만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 64

# 4) 플레이스홀더
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])

# ✅ 드롭아웃 제어(훈련: <1.0, 평가: 1.0)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

# ReLU에 적합한 He 초기화
he = tf.compat.v1.keras.initializers.he_uniform()

# 5) MLP: 64 -> 128 -> 64 -> 10 (은닉 ReLU + Dropout, 출력은 로짓)
w1 = tf.compat.v1.Variable(he(shape=[n_features, 128])); b1 = tf.compat.v1.Variable(tf.zeros([128]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)   # ▶ 훈련 시만 0.9 등으로

w2 = tf.compat.v1.Variable(he(shape=[128, 64])); b2 = tf.compat.v1.Variable(tf.zeros([64]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(he(shape=[64, num_classes])); b3 = tf.compat.v1.Variable(tf.zeros([num_classes]))
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.softmax(logits)

# 6) 손실/옵티마이저 (안정적 softmax CE)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yph, logits=logits))
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 7) 정확도 (argmax 비교)
pred_idx = tf.argmax(probs, axis=1)
true_idx = tf.argmax(yph, axis=1)
acc_tf   = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# 8) 학습 루프(미니배치)
epochs = 800
batch_size = 64
steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(X_train.shape[0])
        X_tr, y_tr = X_train[idx], y_train[idx]

        # ▶ 훈련: 드롭아웃 ON (예: keep_prob=0.9 → 10% 드롭)
        for step in range(steps_per_epoch):
            s = step * batch_size
            e = min((step + 1) * batch_size, X_train.shape[0])
            feed = {x: X_tr[s:e], yph: y_tr[s:e], keep_prob1: 0.9, keep_prob2: 0.9}
            sess.run(train_op, feed_dict=feed)

        if epoch % 100 == 0 or epoch == 1:
            tr_loss, tr_acc = sess.run(
                [loss, acc_tf],
                feed_dict={x: X_train, yph: y_train, keep_prob1: 1.0, keep_prob2: 1.0}  # ▶ 평가: 드롭아웃 OFF
            )
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

    # 9) 최종 평가 (드롭아웃 OFF)
    tr_pred = sess.run(pred_idx, feed_dict={x: X_train, yph: y_train, keep_prob1: 1.0, keep_prob2: 1.0})
    te_pred = sess.run(pred_idx, feed_dict={x: X_test,  yph: y_test,  keep_prob1: 1.0, keep_prob2: 1.0})

# 10) Sklearn 지표
acc_tr = accuracy_score(y_train_int, tr_pred)
acc_te = accuracy_score(y_test_int,  te_pred)
cm_te  = confusion_matrix(y_test_int, te_pred)

print("\n=== Digits (ReLU + Dropout MLP) ===")
print(f"Train Acc: {acc_tr:.4f}")
print(f" Test Acc: {acc_te:.4f}")
print(" Confusion Matrix (test):\n", cm_te)

# === Digits (ReLU + Dropout MLP) ===
# Train Acc: 1.0000
#  Test Acc: 0.9722
#  Confusion Matrix (test):
#  [[36  0  0  0  0  0  0  0  0  0]
#  [ 0 37  0  0  0  0  0  0  0  0]
#  [ 0  0 34  0  0  0  0  0  1  0]
#  [ 0  0  0 35  0  0  0  0  1  1]
#  [ 0  0  0  0 36  0  0  0  0  0]
#  [ 0  0  0  0  0 35  1  0  0  0]
#  [ 0  0  0  0  0  0 36  0  0  0]
#  [ 0  0  0  0  1  0  0 35  0  0]
#  [ 0  2  1  0  0  0  0  0 32  0]
#  [ 0  0  0  0  0  0  0  0  2 34]]