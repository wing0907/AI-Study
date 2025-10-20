# TF1-style Softmax Regression on Iris
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TF2 환경 호환 (그래프 모드)
tf.compat.v1.disable_eager_execution()
np.random.seed(7777)
tf.compat.v1.set_random_seed(7777)

# 1) 데이터 로드
iris = load_iris()
X = iris.data.astype(np.float32)         # (150, 4)
y_int = iris.target                      # (150,)  -> 0,1,2
num_classes = 3

# 원-핫 인코딩
y = np.eye(num_classes, dtype=np.float32)[y_int]  # (150, 3)

# 학습/검증 분리 (계층적 분리)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_int, random_state=7777
)

# 표준화 (수렴 안정)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# 2) 플레이스홀더 & 변수
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random.normal([4, 3]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.zeros([3]), name='bias', dtype=tf.float32)   # ✅ 다중분류이므로 [3]

# 3) 모델
logits = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(logits)

# 3-1) 손실 (안정적인 softmax CE)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# ✅ 다중분류 정확도: argmax 비교
pred_idx = tf.argmax(hypothesis, axis=1)
true_idx = tf.argmax(y_ph, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(pred_idx, true_idx), tf.float32))

# 3-2) 학습
epochs = 2001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val, acc_tr = sess.run(
            [train, loss, acc],
            feed_dict={x: X_train, y_ph: y_train}
        )
        if step % 200 == 0:
            print(f"Step {step:04d} | Train Loss: {loss_val:.4f} | Train Acc(TF): {acc_tr:.4f}")

    # 최종 파라미터
    w_val, b_val = sess.run([w, b])

    # 최종 Train/Test 정확도
    tr_acc = sess.run(acc, feed_dict={x: X_train, y_ph: y_train})
    te_acc = sess.run(acc, feed_dict={x: X_test,  y_ph: y_test})

    print("\nfinal weight:\n", w_val)
    print("bias:\n", b_val)
    print(f"\nTrain Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")


# final weight:
#  [[-1.2936407   1.3688408   0.6704963 ]
#  [ 1.8035504  -0.66021997 -1.0497352 ]
#  [-3.389996   -1.65599     2.2742968 ]
#  [-2.1568441  -0.6946922   4.001658  ]]
# bias:
#  [-0.5638641  2.8491557 -2.285293 ]

# Train Acc: 0.9833 | Test Acc: 0.9333