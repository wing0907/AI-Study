# 07번을 copy해서 08의 3번 InteractiveSession() 과
# .eval() 로 바꿔봐
import tensorflow as tf

# TF2 환경에서 TF1 스타일 사용
tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(333)

# 1. 데이터
x_data = [1, 2, 3, 4, 5]
y_data = [4, 6, 8, 10, 12]
x_test_data = [6, 7, 8]

# 플레이스홀더
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 변수 (가중치, 편향)
w = tf.compat.v1.Variable(tf.random.normal([1]), name="weight")
b = tf.compat.v1.Variable(tf.random.normal([1]), name="bias")

# 모델: y = wx + b
hypothesis = x * w + b

# 손실/최적화
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.09)
train = optimizer.minimize(loss)

# ───────────────────────────────────────────────
# InteractiveSession + .eval() 활용
# ───────────────────────────────────────────────
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

# 훈련
epochs = 100
for step in range(epochs):
    _, loss_val = sess.run([train, loss], feed_dict={x: x_data, y: y_data})
    if step % 10 == 0:
        # .eval()을 사용해 변수 값 확인
        print(step, "loss:", loss_val, "w:", w.eval(), "b:", b.eval())

# ───────────────────────────────────────────────
# 4. 예측 (두 가지 방식 모두 .eval() 사용)
# ───────────────────────────────────────────────

# 1) placeholder 방식 (그래프에서 계산)
x_test_ph = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict_graph = x_test_ph * w + b
preds_graph = y_predict_graph.eval(feed_dict={x_test_ph: x_test_data})
print("[6,7,8] 그래프 방식 예측:", preds_graph)

# 2) 넘파이 방식 (변수 값만 뽑아와 직접 계산)
w_val, b_val = w.eval(), b.eval()
preds_numpy = [float(xi * w_val + b_val) for xi in x_test_data]
print("[6,7,8] 넘파이 방식 예측:", preds_numpy)

sess.close()

# [결과]
# 0 loss: 138.14766 w: [6.2373466] b: [1.125248]
# 10 loss: 1572.22 w: [14.934366] b: [4.078713]
# 20 loss: 17975.797 w: [44.716087] b: [12.727548]
# 30 loss: 205568.92 w: [145.70164] b: [40.992424]
# 40 loss: 2350883.5 w: [487.40582] b: [135.85449]
# 50 loss: 26884708.0 w: [1643.0994] b: [456.12128]
# 60 loss: 307453630.0 w: [5551.4346] b: [1538.7834]
# 70 loss: 3516039700.0 w: [18768.387] b: [5199.754]
# 80 loss: 40209390000.0 w: [63464.402] b: [17579.895]
# 90 loss: 459834460000.0 w: [214613.67] b: [59445.848]
# [6,7,8] 그래프 방식 예측: [-4032981.2 -4675485.  -5317988. ]
# [6,7,8] 넘파이 방식 예측: [-4032981.25, -4675485.0, -5317988.0]