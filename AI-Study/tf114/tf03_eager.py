import tensorflow as tf
print('tf version : ', tf.__version__)
print('즉시실행모드 : ', tf.executing_eagerly())
# tf version :  1.14.0
# 즉시실행모드 :  False         #이게 뭔지는 몰라도 지금 실행이 안되고 있다는 뜻


# 가상환경 변경 (tf114cpu -> tf274cpu)
# tf274cpu 로 변경해서 실행함
# tf version :  2.7.4
# 즉시실행모드 :  True          #1버전에서 2버전으로 업데이트 되면서 '즉시실행모드'가 생겼다는 걸 알 수 있음

tf.compat.v1.disable_eager_execution()      #꺼짐
print('즉시실행모드 : ', tf.executing_eagerly())
# 즉시실행모드 :  False

tf.compat.v1.enable_eager_execution()       #다시켜짐
print('즉시실행모드 : ', tf.executing_eagerly())
# 즉시실행모드 :  True

hello = tf.constant("Hello World!!")
sess = tf.compat.v1.Session()
print(sess.run(hello))                      
# tensorflow 2에서 즉시실행모드가 켜져있지만 sess.run이 빠져있기 때문에 에러가 남
# 즉시실행모드는 그냥 print 하면 출력되는 것과 같음
# 즉시실행모드를 꺼버리면 출력된다. 부득이하게 텐서 1을 사용하게 되면 텐서 2에서 즉시실행모드를 끄고 텐서 1의 값을 출력할 수 있다

######################################################################
# 즉시 실행모드 -> 텐서1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행
# tf.compat.v1.disable_eager_execution() # 즉시 실행 모드 끄기 //텐서플로우 1.0 문법(디폴트)
# tf.compat.v1.enable_eager_execution()  # 즉시 실행 모드 킴  // 텐서플로우 2.0 사용 가능

# sess.run() 실행시!!!
# 가상환경            즉시 실행 모드           사용가능
# 1.14.0              disable (디폴트)        b'Hello world!'
# 1.14.0              enable                 error
# 2.7.4               disable (디폴트)        b'Hello world!'
# 2.7.4               enable                 error

"""
Tensor1 은 '그래프 연산' 모드
Tensor2 는 '즉시 실행' 모드

tf.compat.v1.enable_eager_execution()   # 즉시 실행 모드 킴
-> Tensor 2의 디폴트

tf.compat.v1.disable_eager_execution()  # 즉시 실행 모드 끄기
-> 그래프 연산모드로 돌아감
-> Tensor 1코드를 쓸 수 있음

tf.executing_eagerly()
-> True : 즉시 실행 모드, Tensor 2 코드만 써야함
-> False : 그래프 연산 모드, Tensor 1 코드를 쓸 수 있음
"""