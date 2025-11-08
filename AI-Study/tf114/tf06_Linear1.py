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
# model.fit()
sess = tf.compat.v1.Session()               # session 초기화
sess.run(tf.global_variables_initializer()) # variables 초기화

epochs = 4000
for step in range(epochs):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
# x & y 고정값이 작기 때문에 placeholder에 값을 안넣었는데도 돌아간다. 

sess.close() # 코드가 다 돌고나서 자동으로 session이 닫히지만 코드가 길어지고 데이터가 커지면 항상 session 열면 닫아줘야 문제가 안생긴다.

# 0 0.059999973 0.94 0.36
# 20 0.00568093 0.91246027 0.19899848
# 40 0.002146402 0.9461914 0.12231965
# 60 0.0008109684 0.9669251 0.07518694
# 80 0.00030640562 0.9796697 0.046215598

