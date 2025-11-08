import tensorflow as tf
tf.random.set_random_seed(333) 

#1. 데이터
# x = [1,2,3,4,5]
# y = [4,6,8,10,12]

x = tf.placeholder(tf.float32, shape=[None]) # x 와 w 의 shape가 맞아야 함
y = tf.placeholder(tf.float32, shape=[None])

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
    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:[1,2,3,4,5], y:[4,6,8,10,12]}) 
        if step % 10 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)   

# sess.close() 

# results
# 0 138.14766 [6.2373466] [1.125248]
# 10 1572.22 [14.934366] [4.078713]
# 20 17975.797 [44.716087] [12.727548]
# 30 205568.92 [145.70164] [40.992424]
# 40 2350883.5 [487.40582] [135.85449]
# 50 26884708.0 [1643.0994] [456.12128]
# 60 307453630.0 [5551.4346] [1538.7834]
# 70 3516039700.0 [18768.387] [5199.754]
# 80 40209390000.0 [63464.402] [17579.895]
# 90 459834460000.0 [214613.67] [59445.848]