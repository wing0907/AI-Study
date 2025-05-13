import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)
print(node3) # Tensor("Add:0", shape=(), dtype=float32)
# tensor 형태가 출력되게 됨.

sess = tf.compat.v1.Session() #그래프 연산으로 출력해야 원하는 값이 출력된다. 이렇게 하면 warning 안뜸
print(sess.run(node3))



