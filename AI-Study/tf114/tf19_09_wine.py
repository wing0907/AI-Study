# TF1-style Softmax Regression on Wine (3 classes)
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 그래프 모드 & 시드 고정
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
wine = load_wine()
X = wine.data.astype(np.float32)     # (178, 13)
y_int = wine.target                  # (178,) -> 0,1,2
num_classes = 3
y = np.eye(num_classes, dtype=np.float32)[y_int]  # (178, 3)

# 2) 학습/검증 분리 (계층적)
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y, y_int, test_size=0.2, stratify=y_int, random_state=7777
)

# 3) 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 13

# 4) TF1 그래프 정의
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes])

w = tf.compat.v1.Variable(tf.random.normal([n_features, num_classes]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.zeros([num_classes]), name='bias', dtype=tf.float32)  # 다중분류이므로 [3]

logits = tf.matmul(x, w) + b
probs  = tf.nn.softmax(logits)

# 안정적인 softmax CE
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train_op  = optimizer.minimize(loss)

# 정확도 (argmax 비교)
pred_idx = tf.argmax(probs, axis=1)
true_idx = tf.argmax(y_ph, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# 5) 학습
epochs = 2001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val, acc_tr = sess.run(
            [train_op, loss, acc],
            feed_dict={x: X_train, y_ph: y_train}
        )
        if step % 200 == 0:
            print(f"Step {step:04d} | Train Loss: {loss_val:.4f} | Train Acc(TF): {acc_tr:.4f}")

    # 최종 성능
    tr_acc_tf = sess.run(acc, feed_dict={x: X_train, y_ph: y_train})
    te_acc_tf = sess.run(acc, feed_dict={x: X_test,  y_ph: y_test})

    # sklearn 기준 정확도(테스트)
    y_test_pred = sess.run(pred_idx, feed_dict={x: X_test})
    sk_acc_test = accuracy_score(y_test_int, y_test_pred)

    print("\n=== Results ===")
    print(f"Train Acc (TF): {tr_acc_tf:.4f} | Test Acc (TF): {te_acc_tf:.4f}")
    print(f" Test Acc (sklearn): {sk_acc_test:.4f}")
