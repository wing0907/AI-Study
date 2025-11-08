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
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(train_csv)
# print(train_csv.head())           # 앞부분 5개 디폴트
# print(train_csv.tail())           # 뒷부분 5개
print(train_csv.head(10))           # 앞부분 10개          

print(train_csv.isna().sum())       # train data의 결측치 갯수 확인  -> 없음
print(test_csv.isna().sum())        # test data의 결측치 갯수 확인   -> 없음

print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
    #    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    #    'EstimatedSalary', 'Exited']

#  문자 데이터 수치화!!!
from sklearn.preprocessing import LabelEncoder

le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())         # 잘 나왔는지 확인하기. pandas는 value_counts() 사용
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884

train_csv = train_csv.drop(['CustomerId', 'Surname',], axis=1)  # 2개 이상은 리스트
test_csv = test_csv.drop(['CustomerId', 'Surname', ], axis=1)


x = train_csv.drop(['Exited'], axis=1)
print(x.shape)      # (165034, 10)
y = train_csv['Exited']
print(y.shape)      # (165034,)


from sklearn.preprocessing import StandardScaler

# 1. 컬럼 분리
x_other = x.drop(['EstimatedSalary'], axis=1)
x_salary = x[['EstimatedSalary']]


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
plt.ylim(0.5, 1.01)
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 📌 이상치 제거 전 성능:
# XGBoost (Before) - Accuracy: 0.8619, F1: 0.6282
# LightGBM (Before) - Accuracy: 0.8631, F1: 0.6292
# CatBoost (Before) - Accuracy: 0.8639, F1: 0.6303

# 📌 이상치 제거 후 성능 (EllipticEnvelope):
# XGBoost (After) - Accuracy: 0.8659, F1: 0.6044
# LightGBM (After) - Accuracy: 0.8687, F1: 0.6090
# CatBoost (After) - Accuracy: 0.8675, F1: 0.6050