import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])


w = tf.Variable(tf.random.normal([1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

#2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

################## optimizer ###################
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

# loss = (hypothesis - y)**2 (x * w - y)**2 
# loss를w로미분 = 2 * (x * w - y) * x
# lr = 0.02
# gradient = tf.reduce_mean((x * w - y) * x)
# descent = w - lr * gradient
# train = w.assign(descent)


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

################## 실습 : R2, mae 만들기 #######################
from sklearn.metrics import r2_score, mean_absolute_error   

# 훈련 데이터에 대한 예측값
y_train_pred = [w_val * xi + b_val for xi in x_data]
r2 = r2_score(y_data, y_train_pred)
mae = mean_absolute_error(y_data, y_train_pred)


y_predict = x_test_data * w_val + b_val
r2 = r2_score(y_data, y_predict)
mae = mean_absolute_error(y_data, y_predict)


print('r2_score:', r2)
print('mae:', mae)

# =============== predict ===============
# [6,7,8]결과:  [12.996304 14.994827 16.99335 ]
# r2_score: 0.9999993863760025
# mae: 0.0019202709197998046