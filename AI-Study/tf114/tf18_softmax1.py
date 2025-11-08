import tensorflow as tf
import numpy as np
tf.random.set_random_seed(7777)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7],
          ]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0],
          ]
x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name='weight', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

#3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))  # binary/categorical crossentropy는 바뀔일이 없다. TF1에서는 이렇게 사용하면 됨
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

#3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run(
        [loss, train, w, b, acc],
        feed_dict={x: x_data, y: y_data}
    )
    if step % 200 == 0:
        print(f"Step {step:04d} | Loss: {cost_val:.4f} | Acc(TF): {acc_val:.4f}")

print("\nfinal weight: ", w_val)
print("bias: ", b_val)
print("acc: ", acc_val)

# final weight: [[-4.972028    0.02631865  5.691417  ]
#  [-0.56481296 -0.949926    1.6083387 ]
#  [-0.284065   -0.01248386 -2.4751315 ]
#  [ 1.7273595   1.1471914  -1.724422  ]] 
# bias:  [1.0292974e-06]
# acc:  0.9166667