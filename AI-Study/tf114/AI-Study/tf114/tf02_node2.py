import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈
# 뺄셈 subtract
# 곱셈 multiply
# 나눗셈 divide

node3 = tf.subtract(node1, node2)
sess = tf.Session()
print(sess.run(node3)) # -1.0

node4 = tf.multiply(node1, node2)
node5 = tf.divide(node1, node2)
node6 = tf.add(node1, node2)
print(sess.run(node4)) # 6.0
print(sess.run(node5)) # 0.6666667
print(sess.run(node6)) # 5.0

node33 = node1 - node2   # 이렇게 해도 됨
print(sess.run(node33))