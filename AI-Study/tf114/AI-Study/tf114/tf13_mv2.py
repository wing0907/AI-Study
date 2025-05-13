import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

tf.random.set_random_seed(7777)

# 1. 데이터 
x_data = [[73., 93., 89.],
           [80, 96., 73.],
           [80., 88., 91.],
           [98., 66., 75.],
           [93., 90., 100.]]
y_data  = [[152.], [185.], [180.], [196.], [142.]]

x  = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([3, 1]), name='weights')
b  = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name='bias')

# 2. 모델
# hypothesis = x*w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

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
        _, loss_val, w_val, b_val = sess.run(
            [train, loss, w, b],
            feed_dict={x:x_data, y:y_data}
        )

        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)

        loss_val_list.append(loss_val)

    print("="*15, "최종 가중치", "="*15)
    print("w:", w_val, "b:", b_val)

    # 학습 끝난 후 예측
    y_pred = sess.run(hypothesis, feed_dict={x: x_data})  # (5,1)

    r2  = r2_score(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)
    print("r2_score:", r2)
    print("mae:", mae)
    
# =============== 최종 가중치 ===============
# w: [[ 2.121375 ]
#  [ 1.291767 ]
#  [-1.5121632]] b: [8.069591]
# r2_score: 0.2343449754964254
# mae: 14.532553100585938