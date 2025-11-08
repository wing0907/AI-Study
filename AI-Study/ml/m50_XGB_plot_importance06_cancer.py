import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

# 데이터 로드
data = load_breast_cancer()
x = data.data
y = data.target
feature_names = data.feature_names

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42, stratify=y
)

# 모델 정의
model_dt = DecisionTreeClassifier(random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_gb = GradientBoostingClassifier(random_state=42)
model_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

models = [model_dt, model_rf, model_gb, model_xgb]
titles = ["DecisionTree", "RandomForest", "GradientBoosting", "XGBoost"]

# 학습
for model in models:
    model.fit(x_train, y_train)

# 시각화: feature_importances_ 기반
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, model in enumerate(models):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    axes[i].barh(range(len(importances)), importances[sorted_idx], align='center')
    axes[i].set_yticks(range(len(importances)))
    axes[i].set_yticklabels(np.array(feature_names)[sorted_idx])
    axes[i].set_title(f"{titles[i]} Feature Importances")
    axes[i].set_xlabel("Importance")

plt.tight_layout()
plt.show()
