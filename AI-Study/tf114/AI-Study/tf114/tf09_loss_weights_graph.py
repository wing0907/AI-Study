import tensorflow as tf
import matplotlib.pyplot as plt
tf.random.set_random_seed(777)

# matplotlib==3.1.1
# numpy==1.16.0

x = [1,2,3]
y = [1,2,3]

w = tf.compat.v1.placeholder(tf.float32)

#2. 모델
hypothesis = x * w

#3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))  # mse

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print("="*80)
print(w_history)

print("="*80)
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('Weights')
plt.ylabel('Loss')
plt.grid()
plt.show()


