import tensorflow as tf
tf.random.set_random_seed(333) 

#1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None]) # x 와 w 의 shape가 맞아야 함
y = tf.placeholder(tf.float32, shape=[None])

x_test = [6,7,8]
#[실습] predict 만들어 보기! 예상 값: [14,16,18]
x_test_p = tf.placeholder(tf.float32, shape=[None])

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) # 여기서 1은 shape 이다
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# print(w)        # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

#2. 모델구성
# y = wx + b
hypothesis = x * w + b   # 이게 리니어 리니 아니여 모델임

# 테스트 예측 연산
pred_op = x_test_p * w + b


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
                                             feed_dict={x:x_data, y:y_data}) 
        if step % 10 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)   

    # 훈련 후 예측
    preds = sess.run(pred_op, feed_dict={x_test_p: x_test})
    print("x_test:", x_test)
    print("preds :", preds)  # 기대값: [14, 16, 18] 근처

# sess.close() 

# results
# 0 2.5234306 [3.1144142] [-0.98780894]
# 10 9.081138 [3.5234795] [0.0017938]
# 20 93.191956 [5.5036454] [1.1936216]
# 30 1059.9978 [12.638177] [3.6421044]
# 40 12119.027 [37.086807] [10.760757]
# 50 138591.53 [120.00133] [33.98134]
# 60 1584932.4 [400.5681] [111.880775]
# 70 18125290.0 [1349.4924] [374.8549]
# 80 207280850.0 [4558.5786] [1263.8213]
# 90 2370464500.0 [15410.863] [4269.804]
# x_test: [6, 7, 8]
# preds : [-289549.06 -335677.97 -381806.88]