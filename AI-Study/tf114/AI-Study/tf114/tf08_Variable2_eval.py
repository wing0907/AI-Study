# 07번을 copy해서 08의 2번 .eval(session=sess) 변수초기화 2번으로 바꿔봐

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

    # y_predict = x_test * w_val + b_val
    y_predict = hypothesis.eval(feed_dict={x:x_test}, session=sess)
    print("[6,7,8]결과:", sess.run(y_predict, feed_dict={x_test:x_test_data}))

    # #2. 파이썬(넘파이)방식
    # y_predict2 = x_test_data * w_val + b_val
    # print("[6,7,8]결과: ", y_predict2)

# sess.close()


exit()
























import tensorflow as tf

# TF2 환경에서도 TF1 스타일 사용
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(333)

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]

x = tf.compat.v1.placeholder(tf.float32, shape=[None])  # x 와 w 의 shape가 맞아야 함
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

x_test_data = [6,7,8]   # 예측용 원시 데이터

# 초기 가중치/편향 (shape=[1])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 2. 모델구성: y = w*x + b
hypothesis = x * w + b

# (옵션) 테스트 예측 연산: placeholder로 넣을 때 사용 가능
x_test_p = tf.compat.v1.placeholder(tf.float32, shape=[None])
pred_op = x_test_p * w + b

# 3-1. 컴파일(손실/옵티마이저)
loss = tf.reduce_mean(tf.square(hypothesis - y))      # MSE
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.09)
train = optimizer.minimize(loss)

# 3-2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 100
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run(
            [train, loss, w, b],
            feed_dict={x: x_data, y: y_data}
        )
        if step % 10 == 0:
            print(step, loss_val, w_val, b_val)

    # ─────────────────────────────────────────────
    # 4. 예측 (eval(session=sess) 활용)
    # ─────────────────────────────────────────────

    # 4-1) placeholder 방식 (그래프에서 계산)
    #     변수 w, b 자체를 사용하여 그래프를 유지한 채 계산 → .eval(session=sess)
    x_test_ph = tf.compat.v1.placeholder(tf.float32, shape=[None], name="x_test_ph")
    y_predict_graph = x_test_ph * w + b
    preds_graph = y_predict_graph.eval(session=sess, feed_dict={x_test_ph: x_test_data})
    print("[6,7,8] 그래프 방식 예측:", preds_graph)   # 기대값 근처: [14,16,18]

    # 4-2) 파이썬(넘파이) 방식 (넘파이로 계산)
    #     학습된 변수 값을 넘파이로 뽑아와서 직접 계산 → .eval(session=sess)
    w_np = w.eval(session=sess)   # shape=(1,)
    b_np = b.eval(session=sess)   # shape=(1,)
    preds_numpy = [xi * w_np + b_np for xi in x_test_data]
    # 위는 각 항목별로 (1,) 배열이 나오니 보기 좋게 펼쳐줌
    preds_numpy = [float(p) for p in preds_numpy]
    print("[6,7,8] 넘파이 방식 예측:", preds_numpy)

    # (참고) 기존 pred_op도 eval로 동일하게 사용 가능
    preds_predop = pred_op.eval(session=sess, feed_dict={x_test_p: x_test_data})
    print("[6,7,8] pred_op 예측:", preds_predop)

# [결과]
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
# [6,7,8] 그래프 방식 예측: [-4032981.2 -4675485.  -5317988. ]
# [6,7,8] 넘파이 방식 예측: [-4032981.25, -4675485.0, -5317988.0]
# [6,7,8] pred_op 예측: [-4032981.2 -4675485.  -5317988. ]
