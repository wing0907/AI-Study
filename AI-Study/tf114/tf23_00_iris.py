# TF1-style MLP on Iris (4 -> 16 -> 12 -> 3)
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 그래프 모드 & 시드 고정 (TF2에서도 TF1 방식으로 실행)
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
iris = load_iris()
X = iris.data.astype(np.float32)        # (150, 4)
y_int = iris.target                     # (150,) -> 0,1,2
num_classes = 3
y = np.eye(num_classes, dtype=np.float32)[y_int]  # (150, 3) 원-핫

# 2) 학습/검증 분리(계층적)
X_train, X_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    X, y, y_int, test_size=0.2, stratify=y_int, random_state=7777
)

# 3) 표준화(수렴 안정)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# 4) 플레이스홀더
x   = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
yph = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

# (XOR 때와 동일 컨셉) Glorot(Xavier) 초기화 + tanh 은닉 2층
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

# 4 -> 16
w1 = tf.compat.v1.Variable(glorot(shape=[4, 16]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([16]),        name='b1')
h1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

# 16 -> 12
w2 = tf.compat.v1.Variable(glorot(shape=[16, 12]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([12]),         name='b2')
h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)

# 12 -> 3 (logits에는 활성화 X)
w3 = tf.compat.v1.Variable(glorot(shape=[12, 3]), name='w3')
b3 = tf.compat.v1.Variable(tf.zeros([3]),         name='b3')
logits = tf.matmul(h2, w3) + b3
probs  = tf.nn.softmax(logits)

# 5) 손실/옵티마이저 (안정적 softmax CE)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yph, logits=logits))
train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.03).minimize(loss)

# 6) 정확도 (argmax 비교)
pred_idx = tf.argmax(probs, axis=1)
true_idx = tf.argmax(yph, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# 7) 학습 루프
epochs = 2000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs + 1):
        _, lval, aval = sess.run(
            [train_op, loss, acc],
            feed_dict={x: X_train, yph: y_train}
        )
        if step % 200 == 0:
            print(f"Step {step:04d} | Train Loss: {lval:.4f} | Train Acc(TF): {aval:.4f}")

    # 8) 최종 성능
    tr_acc = sess.run(acc, feed_dict={x: X_train, yph: y_train})
    te_acc = sess.run(acc, feed_dict={x: X_test,  yph: y_test})

    # Sklearn 기준(테스트)
    y_test_pred = sess.run(pred_idx, feed_dict={x: X_test})
    sk_acc_test = accuracy_score(y_test_int, y_test_pred)

    print("\n=== Results ===")
    print(f"Train Acc (TF): {tr_acc:.4f} | Test Acc (TF): {te_acc:.4f}")
    print(f" Test Acc (sklearn): {sk_acc_test:.4f}")

# === Results ===
# Train Acc (TF): 1.0000 | Test Acc (TF): 0.9000
#  Test Acc (sklearn): 0.9000