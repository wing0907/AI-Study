import tensorflow as tf
print(tf.__version__)

# node1 = tf.constant(3.0)
# node2 = tf.constant(3.0)
# node3 = node1 + node2

node1 = tf.compat.v1.placeholder(tf.float32)       # placeholder 할 때는 따로 정의하지 않는다 
node2 = tf.compat.v1.placeholder(tf.float32)
node3 = node1 + node2

sess = tf.compat.v1.Session()
# print(sess.run(node3))
print(sess.run(node3, feed_dict={node1:3, node2:4})) # 7.0

# placeholder가 나오면 feed_dict (a.k.a placeholder 값) 를 넣어줘야 함.
# 연산방식은 그래프 연산 방식.
# input 에 placeholder 값을 넣으면 됨. 그리고 값의 내용은 feed_dict 안에서 정의해주면 됨.

print(sess.run(node3, feed_dict={node1:10, node2:17})) # 27.0

node3_triple = node3 * 3
print(node3_triple) # Tensor("mul:0", dtype=float32)
print(sess.run(node3_triple, feed_dict={node1:3, node2:4})) # 21.0


