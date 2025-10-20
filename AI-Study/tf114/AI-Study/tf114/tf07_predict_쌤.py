import tensorflow as tf
tf.random.set_random_seed(333) 

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None]) # x 와 w 의 shape가 맞아야 함
y = tf.placeholder(tf.float32, shape=[None])

x_test_data = [6,7,8]
#[실습] predict 만들어 보기! 예상 값: [14,16,18]


# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # 여기서 1은 shape 이다
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# print(w)        # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

#2. 모델구성
# y = wx + b
hypothesis = x * w + b   # 이게 리니어 리니 아니여 모델임



#3-1. 컴파일
# model.compile(loss='mse', optimizer='sgd')
# Tensor1에서는 loss 와 optimizer를 각각 정의해줘야 함
loss = tf.reduce_mean(tf.square(hypothesis - y))      # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.09)
train = optimizer.minimize(loss)

#3-2. 훈련
# sess = tf.compat.v1.Session()               # session 초기화
with tf.compat.v1.Session() as sess:          # with 문 사용하면 따로 sess.close()안 해도 자동 close 됨
    sess.run(tf.global_variables_initializer()) # variables 초기화

    # model.fit()
    epochs = 500
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data}) 
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)   


#4. 예측.
    #1. placeholder 방식
    x_test = tf.placeholder(tf.float32, shape=[None])

    y_predict = x_test * w_val + b_val
    print("[6,7,8]결과:", sess.run(y_predict, feed_dict={x_test:x_test_data}))

    #2. 파이썬(넘파이)방식
    y_predict2 = x_test_data * w_val + b_val
    print("[6,7,8]결과: ", y_predict2)

# sess.close()
