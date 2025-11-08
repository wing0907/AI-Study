import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import EllipticEnvelope
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로드
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. 이상치 제거 함수 (EllipticEnvelope 사용)
def remove_outliers_elliptic(x, y, contamination=0.1):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    ee = EllipticEnvelope(contamination=contamination, random_state=42)
    ee.fit(x_scaled)
    mask = ee.predict(x_scaled) == 1
    return x[mask], y[mask]

# 3. 공통 모델 훈련 함수
def train_and_evaluate_model(x, y, label=""):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pf = PolynomialFeatures(degree=2, include_bias=False)
    x_pf = pf.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pf, y, train_size=0.8, random_state=123, stratify=y
    )

    models = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        key = f"{name} {label}"
        results[key] = {"accuracy": acc, "f1": f1}
        print(f"{key} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return results

# 4. 이상치 제거 전 성능
print("\n📌 이상치 제거 전 성능:")
scores_before = train_and_evaluate_model(x, y, label="(Before)")

# 5. 이상치 제거 후 성능 (EllipticEnvelope)
x_clean, y_clean = remove_outliers_elliptic(x, y)
print("\n📌 이상치 제거 후 성능 (EllipticEnvelope):")
scores_after = train_and_evaluate_model(x_clean, y_clean, label="(After)")

# 6. 결과 시각화 (Accuracy + F1 함께 표시)
scores = {**scores_before, **scores_after}
labels = list(scores.keys())
acc_scores = [scores[k]['accuracy'] for k in labels]
f1_scores = [scores[k]['f1'] for k in labels]

x_idx = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(14, 6))
bars1 = plt.bar(x_idx - width/2, acc_scores, width, label='Accuracy', color='skyblue')
bars2 = plt.bar(x_idx + width/2, f1_scores, width, label='F1 Score', color='lightgreen')

# 수치 출력
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.4f}", ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{height:.4f}", ha='center', va='bottom', fontsize=8)

plt.ylabel("Score")
plt.title("이상치 제거 전 vs 후 (EllipticEnvelope) - 분류 모델 성능 (Accuracy & F1)")
plt.xticks(x_idx, labels, rotation=45, ha='right')
plt.ylim(0.85, 1.01)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 📌 이상치 제거 전 성능:
# XGBoost (Before) - Accuracy: 0.9649, F1: 0.9718
# LightGBM (Before) - Accuracy: 0.9737, F1: 0.9790
# CatBoost (Before) - Accuracy: 0.9649, F1: 0.9722

# 📌 이상치 제거 후 성능 (EllipticEnvelope):
# XGBoost (After) - Accuracy: 0.9612, F1: 0.9714
# LightGBM (After) - Accuracy: 0.9612, F1: 0.9714
# CatBoost (After) - Accuracy: 0.9515, F1: 0.9645