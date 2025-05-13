"""
import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)
import numpy as np
from sklearn.metrics import accuracy_score

#1. Data
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

#2. Model
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random_normal([2,3]), name='weights1')
b1 = tf.compat.v1.Variable(tf.zeros([3]), name='bias1')
layer1 = tf.matmul(x, w1) + b1 # (N, 3)
 
w2 = tf.compat.v1.Variable(tf.random_normal([3,4]), name='weights2')
b2 = tf.compat.v1.Variable(tf.zeros([4]), name='bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.random_normal([4,1]), name='weights3')
b3 = tf.compat.v1.Variable(tf.zeros([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)


logits = tf.matmul(layer2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)   # 예측 확률

# 3-1. Compile (수치안정 BCE)
# loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
# train = optimizer.minimize(loss)

# XOR는 GD(1e-3)보다 Adam이 잘 수렴
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

# ✅ 정확도: 그래프에서 바로 계산
predicted = tf.cast(hypothesis > 0.5, tf.float32)
acc_tf = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))


# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

#3-2. Train
# epochs = 101
# for step in range(epochs):
#     cost_val, _, = sess.run([loss, train, ],
#                                          feed_dict={x:x_data, y:y_data})
#     if step % 20 == 0:
#         print(step, 'loss:', cost_val)

# #4. Evaluate, Predict
# y_predict = tf.sigmoid(tf.matmul(tf.cast(x_data, tf.float32), w_val) + b_val)
# y_pred = sess.run(tf.cast(y_predict > 0.5, dtype=tf.float32))

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_pred, y_data)
# print('acc:', acc)

# 3-2. Train
epochs = 5000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs + 1):
        _, lval, aval = sess.run([train, loss, acc_tf],
                                 feed_dict={x: x_data, y: y_data})
        if step % 500 == 0:
            print(f"Step {step:04d} | Loss: {lval:.6f} | Acc(TF): {aval:.4f}")

    # 4. Evaluate, Predict
    y_prob, y_bin, acc_val = sess.run([hypothesis, predicted, acc_tf],
                                      feed_dict={x: x_data, y: y_data})
    print("\nprobs:", np.round(y_prob.flatten(), 4))
    print("pred :", y_bin.flatten().astype(int).tolist())
    print("acc(TF):", float(acc_val))

    # (선택) sklearn로도 확인
    print("acc(sklearn):", accuracy_score(y_data, y_bin))
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

tf.compat.v1.random.set_random_seed(7777)
np.random.seed(7777)

# 1. Data (XOR)
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],    [1],    [1],    [0]], dtype=np.float32)

# 2. Model
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# (선택) Xavier 초기화가 수렴에 도움이 됩니다.
glorot = tf.compat.v1.keras.initializers.glorot_uniform()

w1 = tf.compat.v1.Variable(glorot(shape=[2,3]), name='weights1')
b1 = tf.compat.v1.Variable(tf.zeros([3]),       name='bias1')
# ✅ 활성화 추가 (구조는 그대로 2→3)
layer1 = tf.nn.tanh(tf.matmul(x, w1) + b1)      # relu로 바꿔도 OK

w2 = tf.compat.v1.Variable(glorot(shape=[3,4]), name='weights2')
b2 = tf.compat.v1.Variable(tf.zeros([4]),       name='bias2')
# ✅ 활성화 추가 (3→4)
layer2 = tf.nn.tanh(tf.matmul(layer1, w2) + b2)

w3 = tf.compat.v1.Variable(glorot(shape=[4,1]), name='weights3')
b3 = tf.compat.v1.Variable(tf.zeros([1]),       name='bias3')

logits = tf.matmul(layer2, w3) + b3
hypothesis = tf.nn.sigmoid(logits)

# 3-1. Compile (안정적인 BCE)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02).minimize(loss)  # 0.01~0.05 사이 추천

# 정확도
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
    y_bin, acc_val = sess.run([predicted, acc_tf], feed_dict={x: x_data, y: y_data})
    print("pred :", y_bin.flatten().astype(int).tolist())
    print("acc(TF):", float(acc_val))
    print("acc(sklearn):", accuracy_score(y_data, y_bin))

# pred : [0, 1, 1, 0]
# acc(TF): 1.0
# acc(sklearn): 1.0








