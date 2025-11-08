# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =========== 설정 ===========
DATA_DIR = r"D:\Study25WJ\_data\\kaggle\bank"
TRAIN_CANDIDATES = [
    "train.csv", "bank.csv", "bank-full.csv", "bank_full.csv",
    "bank_marketing.csv", "bankmarketing.csv"
]
RANDOM_SEED = 7777
EPOCHS = 2500
LR = 5e-2     # 범주 원-핫 후 특성 많아질 수 있어 조금 낮춤

# TF1 모드
tf.compat.v1.disable_eager_execution()
tf.compat.v1.random.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

def read_csv_kr(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

def load_dataset(data_dir):
    for name in TRAIN_CANDIDATES:
        fpath = os.path.join(data_dir, name)
        if os.path.exists(fpath):
            print(f"[INFO] 사용 파일: {fpath}")
            return read_csv_kr(fpath)
    # 폴더 내 단일 CSV가 있으면 그걸 사용
    csvs = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if len(csvs) == 1:
        fpath = os.path.join(data_dir, csvs[0])
        print(f"[INFO] 사용 파일(자동): {fpath}")
        return read_csv_kr(fpath)
    raise FileNotFoundError(f"{data_dir}에서 학습용 CSV를 찾지 못했습니다. 후보: {TRAIN_CANDIDATES}")

def guess_target_column(df):
    candidates = ["y", "deposit", "subscribed", "subscription",
                  "TARGET", "target", "label", "Label", "class", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    # 이진(고유값 2개) 컬럼을 뒤에서 앞으로 탐색
    for c in df.columns[::-1]:
        if df[c].nunique(dropna=True) == 2:
            return c
    raise ValueError("타깃 컬럼을 찾을 수 없습니다. (y/deposit/target/label 등 확인)")

def to_binary_series(y: pd.Series) -> pd.Series:
    # 문자열이면 소문자 변환 후 yes/no 등 매핑
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        z = y.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1, "no": 0,
            "y": 1, "n": 0,
            "true": 1, "false": 0,
            "1": 1, "0": 0
        }
        out = z.map(mapping)
        if out.isna().any():
            # 그래도 못맵핑되면 두 고유값을 0/1로 매핑
            uniq = z.dropna().unique().tolist()
            if len(uniq) == 2:
                return z.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
            else:
                raise ValueError(f"타깃이 이진이 아닙니다: {uniq}")
        return out.astype(int)
    else:
        # 숫자형인 경우 0/1만 허용, 아니면 0.5 기준 이진화
        vals = pd.unique(y.dropna())
        if set(vals).issubset({0, 1}):
            return y.astype(int)
        return (y.astype(float) >= 0.5).astype(int)

def preprocess_bank(df: pd.DataFrame, target_col: str):
    # id 계열 제거
    drop_like = {"id", "index", "idx", "rowid", "번호"}
    drop_cols = [c for c in df.columns if c.lower() in drop_like or c == target_col]

    y = to_binary_series(df[target_col]).values.reshape(-1, 1).astype(np.float32)

    X = df.drop(columns=drop_cols)
    # 수치/범주 분리
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # 수치 결측치 중앙값 대체
    X_num = X[num_cols].copy()
    if len(num_cols) > 0:
        X_num = X_num.apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median())
    else:
        X_num = pd.DataFrame(index=X.index)

    # 범주 원-핫 인코딩
    if len(cat_cols) > 0:
        X_cat = pd.get_dummies(X[cat_cols].astype(str).fillna("NA"), drop_first=True)
    else:
        X_cat = pd.DataFrame(index=X.index)

    # 표준화(수치만)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(
            scaler.fit_transform(X_num), columns=num_cols, index=X.index
        ).astype(np.float32)
    else:
        X_num_scaled = X_num

    # 결합
    X_final = pd.concat([X_num_scaled, X_cat.astype(np.float32)], axis=1)
    return X_final.values.astype(np.float32), y, X_final.columns.tolist()

def build_tf_graph(n_features):
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features], name="x")
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="y")

    w = tf.compat.v1.Variable(tf.compat.v1.random_normal([n_features, 1]), name="weights", dtype=tf.float32)
    b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name="bias", dtype=tf.float32)

    logits = tf.compat.v1.matmul(x, w) + b
    hypothesis = tf.compat.v1.sigmoid(logits)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LR)
    train_op = optimizer.minimize(loss)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y_), dtype=tf.float32))
    return x, y_, w, b, hypothesis, loss, train_op, predicted, acc

def main():
    # 1) 데이터 로드
    df = load_dataset(DATA_DIR)

    # 2) 타깃 탐지
    target_col = guess_target_column(df)
    print(f"[INFO] 타깃 컬럼: {target_col}")

    # 3) 전처리(원-핫/스케일링)
    X, y, feat_names = preprocess_bank(df, target_col)
    print(f"[INFO] 샘플 수: {X.shape[0]} | 특징 수(원-핫 후): {X.shape[1]}")

    # 4) 학습/평가 분리
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # 5) TF 그래프
    x, y_, w, b, hypothesis, loss, train_op, predicted, acc = build_tf_graph(X_tr.shape[1])

    # 6) 학습
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(EPOCHS):
            _, loss_val, acc_tr = sess.run(
                [train_op, loss, acc],
                feed_dict={x: X_tr, y_: y_tr}
            )
            if step % 200 == 0:
                print(f"Step {step:04d} | Train Loss: {loss_val:.4f} | Train Acc(TF): {acc_tr:.4f}")

        # 예측
        y_tr_pred_tf = sess.run(predicted, feed_dict={x: X_tr, y_: y_tr})
        y_te_pred_tf = sess.run(predicted, feed_dict={x: X_te, y_: y_te})
        tr_acc_tf = sess.run(acc, feed_dict={x: X_tr, y_: y_tr})
        te_acc_tf = sess.run(acc, feed_dict={x: X_te, y_: y_te})

        # sklearn accuracy
        y_tr_true = y_tr.ravel().astype(int)
        y_te_true = y_te.ravel().astype(int)
        y_tr_pred = y_tr_pred_tf.ravel().astype(int)
        y_te_pred = y_te_pred_tf.ravel().astype(int)

        from sklearn.metrics import accuracy_score
        tr_acc_sk = accuracy_score(y_tr_true, y_tr_pred)
        te_acc_sk = accuracy_score(y_te_true, y_te_pred)

        print("\n=== Results ===")
        print(f"Features: {X_tr.shape[1]} | Train Acc (TF): {tr_acc_tf:.4f} | Test Acc (TF): {te_acc_tf:.4f}")
        print(f"Train Acc (sklearn): {tr_acc_sk:.4f} | Test Acc (sklearn): {te_acc_sk:.4f}")

if __name__ == "__main__":
    main()


# === Results ===
# Features: 2808 | Train Acc (TF): 0.8138 | Test Acc (TF): 0.8113
# Train Acc (sklearn): 0.8138 | Test Acc (sklearn): 0.8113