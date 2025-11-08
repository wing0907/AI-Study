import tensorflow as tf
sess = tf.compat.v1.Session()

# 초기값
a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer() # 전체변수를 초기화 해주는 것
sess.run(init)

print(sess.run(a + b))  # [3.]




