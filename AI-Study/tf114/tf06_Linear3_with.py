import tensorflow as tf

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

#2. 모델구성
# y = wx + b
hypothesis = x * w + b   # 이게 리니어 리니 아니여 모델임
# 23명의 머신이 생긴거임. 실력보다 인맥이 더 중요함


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
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
    # x & y 고정값이 작기 때문에 placeholder에 값을 안넣었는데도 돌아간다. 

# sess.close() 

