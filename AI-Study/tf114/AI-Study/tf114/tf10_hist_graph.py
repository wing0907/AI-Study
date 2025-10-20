import tensorflow as tf
tf.random.set_random_seed(777)
import matplotlib.pyplot as plt

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])


w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

#2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

#3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1001
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                            feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)       
        
        loss_val_list.append(loss_val)    
        w_val_list.append(w_val)

    print("="*15, "predict", "="*15)
    y_predict = x_test * w_val + b_val
    print("[6,7,8]결과: ", sess.run(y_predict, feed_dict={x_test:x_test_data}))

# 0 23.95274 [1.4495096] 0.18367653
# 100 0.018055474 [2.0866475] 0.6871752
# 200 0.0046481444 [2.0439634] 0.8412783
# 300 0.0011966113 [2.0223062] 0.91946745
# 400 0.0003080527 [2.011318] 0.95913917
# 500 7.930105e-05 [2.0057425] 0.97926795
# 600 2.0415137e-05 [2.0029137] 0.9894809
# 700 5.256451e-06 [2.0014782] 0.9946629
# 800 1.3536695e-06 [2.00075] 0.99729156
# 900 3.4854838e-07 [2.0003808] 0.9986256
# 1000 8.9753215e-08 [2.000193] 0.9993027
# 1100 2.305029e-08 [2.000098] 0.99964625
# 1200 5.9148535e-09 [2.0000498] 0.99982053
# 1300 1.5184014e-09 [2.0000253] 0.9999091
# 1400 3.959144e-10 [2.0000129] 0.9999538
# 1500 1.0592203e-10 [2.000007] 0.9999758
# 1600 3.274181e-11 [2.0000038] 0.9999866
# 1700 1.1471002e-11 [2.0000024] 0.99999213
# 1800 1.1471002e-11 [2.0000024] 0.99999213
# 1900 1.1471002e-11 [2.0000024] 0.99999213
# 2000 1.1471002e-11 [2.0000024] 0.99999213
# =============== predict ===============
# [6,7,8]결과:  [13.000007 15.00001  17.000011]
            
print("="*15, "그림그리기", "="*15)
# print(loss_val_list)
# print(w_val_list)

# loss 와 epoch 관계
# plt.plot(loss_val_list)
# plt.show()

# w 와 epoch 관계
# plt.plot(w_val_list)
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('weights')
# plt.show()

# w 와 loss 관계
# plt.plot(w_val_list, loss_val_list)
# plt.grid()
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.show()

# [실습] subplot으로 위 3개의 그래프를 한 페이지에 나오게 수정
# plt.subplot(221)


# 그래프 그리기
plt.figure(figsize=(15, 5))

# 1️⃣ loss vs epoch
plt.subplot(1, 3, 1)
# plt.subplot(2, 3, 1)
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss vs Epoch')

# 2️⃣ w vs epoch
plt.subplot(1, 3, 2)
# plt.subplot(2, 3, 2)
plt.plot(w_val_list)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('weights')
plt.title('Weights vs Epoch')

# 3️⃣ w vs loss
plt.subplot(1, 3, 3)
# plt.subplot(2, 3, 3)
plt.plot(w_val_list, loss_val_list)
plt.grid()
plt.xlabel('weights')
plt.ylabel('loss')
plt.title('Loss vs Weights')

plt.tight_layout()
plt.show()