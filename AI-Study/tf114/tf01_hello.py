import tensorflow as tf
print(tf.__version__) # 1.14.0
# python : 3.7.16

## 텐서플로 설치 오류시 - 250829 현재
# pip install protobuf==3.20
# pip install numpy==1.16

print("hello world")

hello = tf.constant("hello world")
print(hello) # Tensor("Const:0", shape=(), dtype=string)


sess = tf.Session()
print(sess.run(hello)) # b'hello world'  #그래프 연산을 실행시킴
# 텐서 1을 하려면 반드시 session을 만들고 run을 해야한다

print(sess.run(hello).decode("utf-8")) 