import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)

# 1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 손실 함수 (binary_crossentropy)
loss = -tf.reduce_mean(y * tf.compat.v1.log(hypothesis) + (1-y)* tf.compat.v1.log(1-hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# 정확도(acc) 계산 (aaa/bbb 예제와 동일한 방식)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)            # 0/1 예측값
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), tf.float32))  # 정확도 평균

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run(
        [loss, train, w, b, acc], feed_dict={x:x_data, y:y_data}
    )
    if step % 200 == 0:
        print(f"Step {step:04d}, Loss: {cost_val:.4f}, Acc: {acc_val:.4f}")

print("\nfinal weight: ", w_val, "bias: ", b_val)

# 최종 예측 결과 출력
h_val, p_val, acc_val = sess.run([hypothesis, predicted, acc], feed_dict={x:x_data, y:y_data})
print("\n최종 hypothesis:\n", h_val.flatten())
print("최종 predicted:\n", p_val.flatten())
print("정답 y_data:\n", [item for sublist in y_data for item in sublist])
print(f"최종 정확도(Acc): {acc_val:.4f}")

sess.close()


# 최종 hypothesis:
#  [0.00898041 0.11040679 0.17166218 0.8512792  0.9749204  0.99241877]
# 최종 predicted:
#  [0. 0. 0. 1. 1. 1.]
# 정답 y_data:
#  [0, 0, 0, 1, 1, 1]
# 최종 정확도(Acc): 1.0000