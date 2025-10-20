import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# TF1 모드 (TF2 환경일 때 필요)
tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(7777)
np.random.seed(7777)

# 1) 데이터 로드
data = load_breast_cancer()
X = data.data.astype(np.float32)                 # (569, 30)
y = data.target.reshape(-1, 1).astype(np.float32)  # (569, 1), 0/1

# 2) train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=7777
)

# 3) 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

n_features = X_train.shape[1]  # 30

# 4) TF1 그래프
x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

logits = tf.compat.v1.matmul(x, w) + b
hypothesis = tf.compat.v1.sigmoid(logits)

# 수치안정 BCE
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

# ✅ TF 방식의 정확도
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y_), dtype=tf.float32))

# 5) 학습
epochs = 3001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(epochs):
        _, loss_val, acc_tr = sess.run(
            [train, loss, acc],
            feed_dict={x: X_train, y_: y_train}
        )
        if step % 300 == 0:
            print(f"Step {step:04d} | Train Loss: {loss_val:.4f} | Train Acc(TF): {acc_tr:.4f}")

    # 최종 파라미터
    w_val, b_val = sess.run([w, b])

    # ✅ 예측은 각각 개별 sess.run 으로! (리스트 아님)
    train_pred_tf = sess.run(predicted, feed_dict={x: X_train, y_: y_train})
    test_pred_tf  = sess.run(predicted, feed_dict={x: X_test,  y_: y_test})

    # ✅ sklearn accuracy_score
    y_train_true = y_train.ravel().astype(int)
    y_test_true  = y_test.ravel().astype(int)
    y_train_pred = train_pred_tf.ravel().astype(int)
    y_test_pred  = test_pred_tf.ravel().astype(int)

    sk_acc_train = accuracy_score(y_train_true, y_train_pred)
    sk_acc_test  = accuracy_score(y_test_true, y_test_pred)

    # 참고로 TF 기준의 test acc도 보고 싶다면:
    test_acc_tf = sess.run(acc, feed_dict={x: X_test, y_: y_test})

    print("\n=== Results ===")
    print("final weight shape:", w_val.shape, "bias:", b_val.flatten())
    print(f"Train Acc (TF): {acc_tr:.4f}")
    print(f" Test Acc (TF): {test_acc_tf:.4f}")
    print(f"Train Acc (sklearn): {sk_acc_train:.4f}")
    print(f" Test Acc (sklearn): {sk_acc_test:.4f}")


# === Results ===
# final weight shape: (30, 1) bias: [0.18357834]
# Train Acc (TF): 0.9846
#  Test Acc (TF): 0.9825
# Train Acc (sklearn): 0.9846
#  Test Acc (sklearn): 0.9825