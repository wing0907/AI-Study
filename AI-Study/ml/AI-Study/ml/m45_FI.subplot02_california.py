from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─── 1. 데이터 ─────────────────────────────────────────────
seed = 123
random.seed(seed)
np.random.seed(seed)

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed, #stratify=y
)

# ─── 2. 모델 구성 및 학습 ─────────────────────────────────
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRFRegressor(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"===== {model.__class__.__name__} =====")
    print("acc :", acc)
    print("importances:", model.feature_importances_)

# ─── 3. 2×2 서브플롯으로 한 번에 그리기 ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # flatten to 1D array of 4 axes

for ax, model in zip(axes, models):
    imp = model.feature_importances_
    n_features = x.shape[1]
    ax.barh(np.arange(n_features), imp, align='center')
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(datasets.feature_names)
    ax.set_xlabel("Feature Importance")
    ax.set_title(model.__class__.__name__)

plt.tight_layout()
plt.show()
