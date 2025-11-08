import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# (TF2 환경일 경우 그래프 모드로 전환; TF1이면 이미 그래프 모드)
tf.compat.v1.disable_eager_execution()

tf.compat.v1.random.set_random_seed(7777)
np.random.seed(7777)

# 1) Data (XOR)
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

# 2) Placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# ✅ 드랍아웃 비율을 제어할 placeholder (훈련: 0.9, 평가: 1.0)
#   - keep_prob: 뉴런을 "유지"할 확률 (즉, 0.9면 10% 드랍)
keep_prob1 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob1")
keep_prob2 = tf.compat.v1.placeholder_with_default(1.0, shape=(), name="keep_prob2")

# (선택) Xavier 초기화: 수렴에 도움
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

# 3) Model (구조 2→3→4→1 유지)
w1 = tf.compat.v1.Variable(glorot(shape=[2,3]), name='weights1')
b1 = tf.compat.v1.Variable(tf.zeros([3]),       name='bias1')
z1 = tf.matmul(x, w1) + b1
h1 = tf.nn.tanh(z1)  # ▶ 비선형 활성화로 XOR 학습 가능

# ✅ 드랍아웃: 훈련중에만 뉴런 일부 비활성화 (평가/예측시엔 keep_prob=1.0로 끔)
#   * TF1은 keep_prob 인자를 사용 (TF2의 rate와 혼용 X)
h1 = tf.nn.dropout(h1, keep_prob=keep_prob1)

w2 = tf.compat.v1.Variable(glorot(shape=[3,4]), name='weights2')
b2 = tf.compat.v1.Variable(tf.zeros([4]),       name='bias2')
z2 = tf.matmul(h1, w2) + b2
h2 = tf.nn.tanh(z2)

# ✅ 두 번째 은닉층에도 드랍아웃 적용 (선택)
h2 = tf.nn.dropout(h2, keep_prob=keep_prob2)

w3 = tf.compat.v1.Variable(glorot(shape=[4,1]), name='weights3')
b3 = tf.compat.v1.Variable(tf.zeros([1]),       name='bias3')
logits = tf.matmul(h2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)  # 예측 확률

# 4) Loss/Opt (수치 안정적인 BCE)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02).minimize(loss)

# 5) Metrics
predicted = tf.cast(hypothesis > 0.5, tf.float32)
acc_tf = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))

# 6) Train
epochs = 5000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs + 1):
        # ▶ 훈련 시엔 드랍아웃 ON (예: 0.9 유지 → 10% 드랍)
        _, lval, aval = sess.run(
            [train, loss, acc_tf],
            feed_dict={x: x_data, y: y_data, keep_prob1: 0.9, keep_prob2: 0.9}
        )
        if step % 500 == 0:
            print(f"Step {step:04d} | Loss: {lval:.6f} | Acc(TF): {aval:.4f}")

    # 7) Evaluate/Predict
    # ▶ 평가/예측 시엔 드랍아웃 OFF (keep_prob=1.0)
    y_bin, acc_val = sess.run(
        [predicted, acc_tf],
        feed_dict={x: x_data, y: y_data, keep_prob1: 1.0, keep_prob2: 1.0}
    )
    print("\npred :", y_bin.flatten().astype(int).tolist())
    print("acc(TF):", float(acc_val))
    print("acc(sklearn):", accuracy_score(y_data, y_bin))

# pred : [0, 1, 1, 0]
# acc(TF): 1.0
# acc(sklearn): 1.0

