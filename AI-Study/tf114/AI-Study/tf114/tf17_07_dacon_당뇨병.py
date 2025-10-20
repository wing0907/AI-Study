import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ===== 설정 =====
DATA_DIR = r"D:\Study25WJ\_data\dacon\diabetes"
TRAIN_CANDIDATES = ["train.csv", "diabetes.csv"]  # 우선순위대로 시도
RANDOM_SEED = 7777
EPOCHS = 3001
LR = 1e-1

# TF1 모드
tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

def read_csv_kr(path):
    """한글 경로/인코딩 대응 CSV 로더"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def load_dataset(data_dir):
    """data_dir에서 train 후보 파일을 찾아 로드"""
    for name in TRAIN_CANDIDATES:
        fpath = os.path.join(data_dir, name)
        if os.path.exists(fpath):
            print(f"[INFO] 사용 파일: {fpath}")
            return read_csv_kr(fpath)
    raise FileNotFoundError(
        f"'{data_dir}'에서 {TRAIN_CANDIDATES} 중 하나를 찾지 못했습니다."
    )

def guess_target_column(df):
    """타깃 컬럼 자동 추정"""
    candidates = ["Outcome", "outcome", "TARGET", "target", "label", "Label", "y", "Y", "class", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    # 후보가 없으면 '이진(고유값 2개)' 컬럼을 뒤에서 앞으로 탐색
    for c in df.columns[::-1]:
        if df[c].nunique(dropna=True) == 2:
            return c
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (Outcome/target/label 등 확인)")

def to_numeric_df(df):
    """모든 열 숫자 변환 시도(문자→숫자), 실패는 NaN -> 이후 중앙값 대체"""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def preprocess(df, target_col):
    """전처리: ID계열 드랍, 숫자화, 결측치 중앙값 대체, 스케일링"""
    # id/번호 계열 컬럼 제거
    drop_like = {"id", "index", "idx", "patientid", "pid", "일련번호", "번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like]
    feat_cols = [c for c in df.columns if c not in drop_cols + [target_col]]

    X_raw = df[feat_cols].copy()
    y_raw = df[target_col].copy()

    # y를 0/1로 변환
    if y_raw.dtype.kind not in "biufc":   # 숫자가 아니면
        uniq = y_raw.dropna().unique().tolist()
        if len(uniq) != 2:
            raise ValueError(f"타깃 컬럼 '{target_col}'이 이진이 아닙니다: {uniq}")
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y_raw = y_raw.map(mapping)
    else:
        # 실수형일 때 0/1이 아닌 값이 섞여 있으면 0/1로 이진화(0.5 기준)
        u = sorted(pd.unique(y_raw.dropna()))
        if not set(u).issubset({0, 1}):
            y_raw = (y_raw >= 0.5).astype(int)

    # X 숫자화 + 결측치 중앙값 대체
    X_num = to_numeric_df(X_raw)
    X_num = X_num.fillna(X_num.median())

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num).astype(np.float32)

    y_arr = y_raw.values.reshape(-1, 1).astype(np.float32)

    return X_scaled, y_arr, feat_cols

def build_tf_graph(n_features):
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features], name="x")
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="y")

    w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1]), name="weights", dtype=tf.float32)
    b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name="bias", dtype=tf.float32)

    logits = tf.compat.v1.matmul(x, w) + b
    hypothesis = tf.compat.v1.sigmoid(logits)

    # 수치안정 BCE
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LR)
    train_op = optimizer.minimize(loss)

    # TF 방식 정확도
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y_), dtype=tf.float32))

    return x, y_, w, b, logits, hypothesis, loss, train_op, predicted, acc

def main():
    # 1) 데이터 로드
    df = load_dataset(DATA_DIR)

    # 2) 타깃 추정
    target_col = guess_target_column(df)
    print(f"[INFO] 타깃 컬럼: {target_col}")

    # 3) 전처리
    X, y, feat_cols = preprocess(df, target_col)
    print(f"[INFO] 특징 수: {X.shape[1]}, 표본 수: {X.shape[0]}")

    # 4) train/val 분리
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    n_features = X_tr.shape[1]

    # 5) TF 그래프
    x, y_, w, b, logits, hypothesis, loss, train_op, predicted, acc = build_tf_graph(n_features)

    # 6) 학습
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(EPOCHS):
            _, loss_val, acc_tr = sess.run(
                [train_op, loss, acc],
                feed_dict={x: X_tr, y_: y_tr}
            )
            if step % 300 == 0:
                print(f"Step {step:04d} | Train Loss: {loss_val:.4f} | Train Acc(TF): {acc_tr:.4f}")

        # 최종 파라미터
        w_val, b_val = sess.run([w, b])

        # 예측 (개별 run으로 받아서 numpy 배열 유지)
        y_tr_pred_tf = sess.run(predicted, feed_dict={x: X_tr, y_: y_tr})
        y_te_pred_tf = sess.run(predicted, feed_dict={x: X_te, y_: y_te})

        # TF 방식 Acc
        tr_acc_tf = sess.run(acc, feed_dict={x: X_tr, y_: y_tr})
        te_acc_tf = sess.run(acc, feed_dict={x: X_te, y_: y_te})

        # sklearn Acc
        y_tr_true = y_tr.ravel().astype(int)
        y_te_true = y_te.ravel().astype(int)
        y_tr_pred = y_tr_pred_tf.ravel().astype(int)
        y_te_pred = y_te_pred_tf.ravel().astype(int)

        tr_acc_sk = accuracy_score(y_tr_true, y_tr_pred)
        te_acc_sk = accuracy_score(y_te_true, y_te_pred)

        print("\n=== Results ===")
        print("final weight shape:", w_val.shape, "| bias:", b_val.flatten())
        print(f"Train Acc (TF): {tr_acc_tf:.4f} | Test Acc (TF): {te_acc_tf:.4f}")
        print(f"Train Acc (sklearn): {tr_acc_sk:.4f} | Test Acc (sklearn): {te_acc_sk:.4f}")

if __name__ == "__main__":
    main()

# === Results ===
# final weight shape: (8, 1) | bias: [-0.8540038]
# Train Acc (TF): 0.7735 | Test Acc (TF): 0.8015
# Train Acc (sklearn): 0.7735 | Test Acc (sklearn): 0.8015