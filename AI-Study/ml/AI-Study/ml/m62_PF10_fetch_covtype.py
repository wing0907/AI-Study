import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 시드 고정 및 설정
np.random.seed(123)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로드
data = fetch_covtype()
x = data.data
y = data.target -1

# 2. 스케일링 + 다항 특성 확장
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x_scaled)   # 핵심 적용

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, train_size=0.8, random_state=123, stratify=y
)

# 4. 모델 정의
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=123, use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=123),
    "CatBoost": CatBoostClassifier(n_estimators=100, random_state=123, verbose=0)
}

# 5. 훈련 및 평가
acc_scores = {}

for name, model in models.items():
    print(f">>> {name} 학습 중...")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    acc_scores[name] = acc
    print(f"{name} 정확도: {acc:.4f}")

# 6. 시각화
plt.figure(figsize=(10, 6))
colors = ['skyblue', 'orange', 'green', 'red']
plt.bar(acc_scores.keys(), acc_scores.values(), color=colors)
plt.ylabel("정확도 (Accuracy)")
plt.title("PolynomialFeatures + 트리 기반 모델 성능 비교 (covtype)")
plt.ylim(0.6, 1.0)
for i, v in enumerate(acc_scores.values()):
    plt.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom')
plt.grid(axis='y')
plt.show()


# RandomForest 정확도: 0.9699
# LightGBM 정확도: 0.8819
# CatBoost 정확도: 0.8560
# XGBoost 정확도: 0.9053