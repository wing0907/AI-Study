import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score

# 시드 고정
np.random.seed(123)

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로드 및 스케일링
data = fetch_california_housing()
x = data.data
y = data.target
feature_names = data.feature_names

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 2. 다항 특성 생성
pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x_scaled)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, train_size=0.8, random_state=123
)

# 4. 모델 정의
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
}

# 5. 훈련 및 평가
r2_scores = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores[name] = r2
    print(f"{name} R² score: {round(r2, 4)}")

# 6. 시각화
plt.figure(figsize=(10, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color=['skyblue', 'orange', 'green', 'red'])
plt.ylabel("R² score")
plt.title("정규화 선형 회귀 모델 성능 비교")
plt.ylim(0.5, 0.85)
for i, v in enumerate(r2_scores.values()):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.grid(axis='y')
plt.show()
