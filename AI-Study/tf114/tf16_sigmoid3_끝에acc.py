import tensorflow as tf
from sklearn.metrics import accuracy_score

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

# 3-1. 손실 (binary cross-entropy)
loss = -tf.reduce_mean(y * tf.compat.v1.log(hypothesis) + (1 - y) * tf.compat.v1.log(1 - hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# ✅ TF 방식의 정확도(acc) 계산
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run(
        [loss, train, w, b, acc],
        feed_dict={x: x_data, y: y_data}
    )
    if step % 200 == 0:
        print(f"Step {step:04d} | Loss: {cost_val:.4f} | Acc(TF): {acc_val:.4f}")

print("\nfinal weight: ", w_val, "bias: ", b_val)

# 최종 예측/정확도 출력
h_val, p_val, acc_val = sess.run([hypothesis, predicted, acc], feed_dict={x: x_data, y: y_data})
print("\n최종 hypothesis:\n", h_val.flatten())
print("최종 predicted:\n", p_val.flatten())
print("정답 y_data:\n", [item for sublist in y_data for item in sublist])
print(f"최종 정확도(Acc, TF): {acc_val:.4f}")

# ✅ sklearn accuracy_score 적용 (끝부분)
y_true = [item for sublist in y_data for item in sublist]     # 1D로 평탄화
y_pred = p_val.flatten().tolist()                              # 0.0/1.0 형태
sk_acc = accuracy_score(y_true, y_pred)
print(f"최종 정확도(Acc, sklearn): {sk_acc:.4f}")

sess.close()


# 최종 hypothesis:
#  [0.00898041 0.11040679 0.17166218 0.8512792  0.9749204  0.99241877]
# 최종 predicted:
#  [0. 0. 0. 1. 1. 1.]
# 정답 y_data:
#  [0, 0, 0, 1, 1, 1]
# 최종 정확도(Acc, TF): 1.0000
# 최종 정확도(Acc, sklearn): 1.0000