# TF1-style MLP on Breast Cancer (ReLU + Dropout)
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import math

# 그래프 모드 & 시드
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
data = load_breast_cancer()
X = data.data.astype(np.float32)            # (569, 30)
y = data.target.reshape(-1, 1).astype(np.float32)  # (569, 1), 0/1

# 2) 학습/검증 분리(계층적)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7777
)

# 3) 표준화(입력만)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]

# 4) 플레이스홀더
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# ✅ 드롭아웃 제어(훈련: <1.0, 평가: 1.0)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

# ReLU에 적합한 He 초기화
he = tf.compat.v1.keras.initializers.he_uniform()

# 5) MLP: 30 -> 128 -> 64 -> 1 (은닉 ReLU + Dropout, 출력은 로짓)
w1 = tf.compat.v1.Variable(he(shape=[n_features, 128])); b1 = tf.compat.v1.Variable(tf.zeros([128]))
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)   # ▶ 훈련 때만 0.9 등으로

w2 = tf.compat.v1.Variable(he(shape=[128, 64])); b2 = tf.compat.v1.Variable(tf.zeros([64]))
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(he(shape=[64, 1])); b3 = tf.compat.v1.Variable(tf.zeros([1]))
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.sigmoid(logits)  # 예측 확률

# 6) 손실/옵티마이저 (이진 분류용 BCE: 수치 안정)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yph, logits=logits))
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

# 7) 정확도(TF)
pred_bin = tf.cast(probs > 0.5, tf.float32)
acc_tf   = tf.reduce_mean(tf.cast(tf.equal(pred_bin, yph), tf.float32))

# 8) 학습 루프
epochs = 600
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
            feed = {x: X_tr[s:e], yph: y_tr[s:e], keep_prob1: 0.9, keep_prob2: 0.9}  # ▶ 드롭아웃 ON
            sess.run(train_op, feed_dict=feed)

        if epoch % 50 == 0 or epoch == 1:
            tr_loss, tr_acc = sess.run(
                [loss, acc_tf],
                feed_dict={x: X_train, yph: y_train, keep_prob1: 1.0, keep_prob2: 1.0}  # ▶ 드롭아웃 OFF
            )
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

    # 9) 최종 평가 (드롭아웃 OFF)
    p_train, b_train = sess.run([probs, pred_bin], feed_dict={x: X_train, yph: y_train})
    p_test,  b_test  = sess.run([probs, pred_bin],  feed_dict={x: X_test,  yph: y_test})

# 10) Sklearn 지표
acc_tr = accuracy_score(y_train, b_train)
acc_te = accuracy_score(y_test,  b_test)
auc_te = roc_auc_score(y_test, p_test)  # 확률로 AUC
cm_te  = confusion_matrix(y_test, b_test)

print("\n=== Breast Cancer (ReLU + Dropout MLP) ===")
print(f"Train Acc: {acc_tr:.4f}")
print(f" Test Acc: {acc_te:.4f} | ROC-AUC: {auc_te:.4f}")
print(" Confusion Matrix (test):\n", cm_te)

# === Breast Cancer (ReLU + Dropout MLP) ===
# Train Acc: 1.0000
#  Test Acc: 0.9825 | ROC-AUC: 0.9977
#  Confusion Matrix (test):
#  [[41  1]
#  [ 1 71]]