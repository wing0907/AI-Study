import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.covariance import EllipticEnvelope
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로드
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
feature_names = x.columns.tolist()

# 2. 이상치 제거 함수 (EllipticEnvelope)
def remove_outliers_elliptic(x, y, contamination=0.1):
    # 스케일링 (EllipticEnvelope은 스케일 민감)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    envelope = EllipticEnvelope(contamination=contamination, random_state=42)
    envelope.fit(x_scaled)
    results = envelope.predict(x_scaled)

    mask = results == 1  # 정상값은 1, 이상치는 -1
    return x[mask], y[mask]

# 3. 공통 훈련 함수
def train_and_evaluate_model(x, y, label=""):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pf = PolynomialFeatures(degree=2, include_bias=False)
    x_pf = pf.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pf, y, train_size=0.8, random_state=123
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    }

    r2_scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores[f"{name} {label}"] = r2
        print(f"{name} {label} R² score: {round(r2, 4)}")
    return r2_scores

# 4. 이상치 제거 전
print("\n📌 이상치 제거 전 성능:")
r2_before = train_and_evaluate_model(x, y, label="(Before)")

# 5. 이상치 제거 후 (EllipticEnvelope)
x_clean, y_clean = remove_outliers_elliptic(x, y, contamination=0.1)
print("\n📌 이상치 제거 후 성능 (EllipticEnvelope):")
r2_after = train_and_evaluate_model(x_clean, y_clean, label="(After)")

# 6. 시각화
r2_scores = {**r2_before, **r2_after}
plt.figure(figsize=(12, 6))
colors = ['skyblue' if 'Before' in name else 'lightgreen' for name in r2_scores.keys()]
plt.bar(r2_scores.keys(), r2_scores.values(), color=colors)
plt.ylabel("R² score")
plt.title("이상치 제거 전 vs 후 - 회귀 모델 성능 (EllipticEnvelope)")
plt.ylim(0.1, 0.85)
plt.xticks(rotation=45)
for i, (name, value) in enumerate(r2_scores.items()):
    plt.text(i, value + 0.005, f"{value:.4f}", ha='center', va='bottom', fontsize=9)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 📌 이상치 제거 전 성능:
# LinearRegression (Before) R² score: 0.5692
# Ridge (Before) R² score: 0.578
# Lasso (Before) R² score: 0.585
# ElasticNet (Before) R² score: 0.5828

# 📌 이상치 제거 후 성능 (EllipticEnvelope):
# LinearRegression (After) R² score: 0.3646
# Ridge (After) R² score: 0.3853
# Lasso (After) R² score: 0.3868
# ElasticNet (After) R² score: 0.3943