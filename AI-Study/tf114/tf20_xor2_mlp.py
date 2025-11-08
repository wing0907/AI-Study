import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# TF1 그래프 모드(TF2 환경 호환)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(7777)
np.random.seed(7777)

# 1. Data (XOR)
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 2. MLP 모델 구성 (2개의 은닉층)
n_hidden1 = 8
n_hidden2 = 8
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

w1 = tf.compat.v1.Variable(glorot(shape=[2, n_hidden1]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([n_hidden1]), name='b1')
h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.compat.v1.Variable(glorot(shape=[n_hidden1, n_hidden2]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([n_hidden2]), name='b2')
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

w3 = tf.compat.v1.Variable(glorot(shape=[n_hidden2, 1]), name='w3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='b3')

logits = tf.matmul(h2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)

# 3-1. Compile (안정적인 BCE)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

# 정확도(0/1 비교)
predicted = tf.cast(hypothesis > 0.5, tf.float32)
acc_tf = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))

# 3-2. Train
epochs = 5000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs + 1):
        _, lval, aval = sess.run([train, loss, acc_tf], feed_dict={x: x_data, y: y_data})
        if step % 500 == 0:
            print(f"Step {step:04d} | Loss: {lval:.6f} | Acc(TF): {aval:.4f}")

    # 4. Evaluate, Predict
    y_pred = sess.run(predicted, feed_dict={x: x_data})
    sk_acc = accuracy_score(y_data.astype(int), y_pred.astype(int))
    print("\nPred:", y_pred.flatten().astype(int).tolist())
    print("acc (sklearn):", sk_acc)

# Step 0000 | Loss: 0.781825 | Acc(TF): 0.5000
# Step 0500 | Loss: 0.001498 | Acc(TF): 1.0000
# Step 1000 | Loss: 0.000488 | Acc(TF): 1.0000
# Step 1500 | Loss: 0.000243 | Acc(TF): 1.0000
# Step 2000 | Loss: 0.000143 | Acc(TF): 1.0000
# Step 2500 | Loss: 0.000093 | Acc(TF): 1.0000
# Step 3000 | Loss: 0.000063 | Acc(TF): 1.0000
# Step 3500 | Loss: 0.000045 | Acc(TF): 1.0000
# Step 4000 | Loss: 0.000032 | Acc(TF): 1.0000
# Step 4500 | Loss: 0.000024 | Acc(TF): 1.0000
# Step 5000 | Loss: 0.000018 | Acc(TF): 1.0000

# Pred: [0, 1, 1, 0]
# acc (sklearn): 1.0
