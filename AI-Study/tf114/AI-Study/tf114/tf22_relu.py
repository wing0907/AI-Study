import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)
np.random.seed(7777)

# 1) Data (XOR)
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

# 2) Placeholders
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# Xavier(Glorot) 초기화: tanh에 잘 맞음
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

# 3) Model (2→3→4→1)
w1 = tf.compat.v1.Variable(glorot(shape=[2,3]), name='w1'); b1 = tf.compat.v1.Variable(tf.zeros([3]), name='b1')
h1 = tf.nn.tanh(tf.matmul(x, w1) + b1)   # ✅ 은닉층 활성화(비선형)

w2 = tf.compat.v1.Variable(glorot(shape=[3,4]), name='w2'); b2 = tf.compat.v1.Variable(tf.zeros([4]), name='b2')
h2 = tf.nn.tanh(tf.matmul(h1, w2) + b2)  # ✅ 은닉층 활성화(비선형)

w3 = tf.compat.v1.Variable(glorot(shape=[4,1]), name='w3'); b3 = tf.compat.v1.Variable(tf.zeros([1]), name='b3')
logits = tf.matmul(h2, w3) + b3          # ❗ 출력층은 활성화 금지(생 로짓)
hypothesis = tf.nn.sigmoid(logits)       # 확률

# 4) Loss/Opt
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

# 5) Metrics
predicted = tf.cast(hypothesis > 0.5, tf.float32)
acc_tf = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))

# 6) Train
epochs = 5000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(epochs + 1):
        _, lval, aval = sess.run([train, loss, acc_tf], feed_dict={x: x_data, y: y_data})
        if step % 500 == 0:
            print(f"Step {step:04d} | Loss: {lval:.6f} | Acc(TF): {aval:.4f}")

    y_bin, acc_val = sess.run([predicted, acc_tf], feed_dict={x: x_data, y: y_data})
    print("\npred :", y_bin.flatten().astype(int).tolist())
    print("acc(TF):", float(acc_val))
    print("acc(sklearn):", accuracy_score(y_data, y_bin))

# pred : [0, 1, 1, 0]
# acc(TF): 1.0
# acc(sklearn): 1.0