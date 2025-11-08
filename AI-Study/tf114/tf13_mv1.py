import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

tf.random.set_random_seed(7777)

# 1. 데이터 
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data  = [152., 185., 180., 196., 142.]

x1 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y  = tf.compat.v1.placeholder(tf.float32, shape=[None])

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))
b  = tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

# 2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# 3-2. 훈련
loss_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 1001
    for step in range(epochs+1):
        _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run(
            [train, loss, w1, w2, w3, b],
            feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data}
        )

        if step % 100 == 0:
            print(step, loss_val, w1_val, w2_val, w3_val, b_val)

        loss_val_list.append(loss_val)

    print("="*15, "최종 가중치", "="*15)
    print("w1:", w1_val, "w2:", w2_val, "w3:", w3_val, "b:", b_val)

    # 훈련 데이터 예측값
    y_pred = [w1_val*xi1 + w2_val*xi2 + w3_val*xi3 + b_val 
              for xi1, xi2, xi3 in zip(x1_data, x2_data, x3_data)]

    # R², MAE 계산
    r2  = r2_score(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)

    print("r2_score:", r2)
    print("mae:", mae)

# =============== 최종 가중치 ===============
# w1: [1.0703661] w2: [0.5407383] w3: [0.39592808] b: [0.5937984]
# r2_score: 0.9996122413632698
# mae: 0.33377685546875