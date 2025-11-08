import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

RANDOM_STATE = 333
TEST_SIZE = 0.2

# ─────────────────────────────────────────
# 1) 데이터 로더들 (각각 X, y, pre_scaled 반환)
# ─────────────────────────────────────────
def _load_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return X, y, False

def _load_wine():
    X, y = load_wine(return_X_y=True)
    return X, y, False

def _load_digits():
    X, y = load_digits(return_X_y=True)
    return X, y, False

def _load_dacon_diabetes():
    path = r'C:\Study25\_data\dacon\diabetes\\'
    train_csv = pd.read_csv(path + 'train.csv', index_col=0)

    X = train_csv.drop('Outcome', axis=1)
    y = train_csv['Outcome']

    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[zero_not_allowed] = X[zero_not_allowed].replace(0, np.nan)
    X = X.fillna(X.mean())

    return X.values, y.values, False  # pre_scaled 아님

def _load_kaggle_bank():
    path = 'C:\Study25\_data\kaggle\\bank\\'
    train_csv = pd.read_csv(path + 'train.csv', index_col=0)

    # 문자형 → 라벨인코딩
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
    train_csv['Gender']    = le_gender.fit_transform(train_csv['Gender'])

    # 불필요 열 제거
    train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)

    X = train_csv.drop(['Exited'], axis=1)
    y = train_csv['Exited']

    # EstimatedSalary만 분리 스케일링 후 다시 합치기 (요청 전처리 준수)
    from sklearn.preprocessing import StandardScaler
    x_other  = X.drop(['EstimatedSalary'], axis=1)
    x_salary = X[['EstimatedSalary']]
    scaler_other  = StandardScaler()
    scaler_salary = StandardScaler()
    x_other_scaled  = scaler_other.fit_transform(x_other)
    x_salary_scaled = scaler_salary.fit_transform(x_salary)
    X_scaled = np.concatenate([x_other_scaled, x_salary_scaled], axis=1)

    # 이 데이터셋은 이미 스케일링을 마쳤으므로 pre_scaled=True
    return X_scaled, y.values, True

# 매핑
LOADERS = {
    "cancer": _load_cancer,
    "dacon_diabetes": _load_dacon_diabetes,
    "kaggle_bank": _load_kaggle_bank,
    "wine": _load_wine,
    "digits": _load_digits,
}

def load_dataset(name: str):
    name = name.lower()
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS)}")
    return LOADERS[name]()

# ─────────────────────────────────────────
# 2) 스케일러 / PCA / 모델
# ─────────────────────────────────────────
scalers = {
    "None": None,
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
}

pca_options = {
    "NoPCA": None,
    "PCA(0.95)": PCA(n_components=0.95, random_state=RANDOM_STATE),
}

models = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVC": SVC(),
    "LogReg": LogisticRegression(
        max_iter=5000, solver="lbfgs", C=1.0, random_state=RANDOM_STATE
    ),
}

# 스케일이 필요한 분류 모델들
NEEDS_SCALING = {"SVC", "LogReg"}

datasets = [
    ("cancer", "이진분류"),
    ("dacon_diabetes", "이진분류"),
    ("kaggle_bank", "이진분류"),   # pre_scaled=True
    ("wine", "다중분류"),
    ("digits", "다중분류"),
]

# ─────────────────────────────────────────
# 3) 실행
# ─────────────────────────────────────────
rows = []

for ds_name, task_type in datasets:
    X, y, pre_scaled = load_dataset(ds_name)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    for scaler_name, scaler in scalers.items():
        # 이미 pre_scaled 데이터라면 추가 스케일링은 생략
        if pre_scaled and scaler is not None:
            continue

        for pca_name, pca_obj in pca_options.items():
            for model_name, model in models.items():
                # 스케일 필요한 모델인데 (pre_scaled가 아니고) 스케일러도 없으면 건너뜀
                if (model_name in NEEDS_SCALING) and (not pre_scaled) and (scaler is None):
                    continue

                steps = []
                # pre_scaled가 False일 때만 스케일러 적용
                if (not pre_scaled) and (scaler is not None):
                    steps.append(scaler)
                if pca_obj is not None:
                    steps.append(pca_obj)
                steps.append(model)

                estimator = make_pipeline(*steps) if steps else model

                # 파이프 설명 문자열
                parts = []
                if (not pre_scaled) and (scaler is not None):
                    parts.append(scaler_name)
                if pca_obj is not None:
                    parts.append(pca_name)
                parts.append(model_name)
                pipeline_desc = " -> ".join(parts) if parts else model_name

                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(x_test)

                acc = accuracy_score(y_test, y_pred)
                f1m = f1_score(y_test, y_pred, average="macro")

                rows.append({
                    "dataset": ds_name,
                    "task": task_type,
                    "pipeline": pipeline_desc if pipeline_desc else model_name,
                    "accuracy": acc,
                    "f1_macro": f1m,
                })

df = pd.DataFrame(rows)
df["accuracy"] = df["accuracy"].round(5)
df["f1_macro"] = df["f1_macro"].round(5)

print("=== 전체 조합 결과 (분류) ===")
print(df.sort_values(by=["dataset", "accuracy"], ascending=[True, False]).to_string(index=False))

# 최고 조합: accuracy ↓, f1_macro ↓
best_cls = df.sort_values(
    by=["dataset", "accuracy", "f1_macro"],
    ascending=[True, False, False]
).groupby("dataset", as_index=False).head(1).reset_index(drop=True)

print("\n=== 데이터셋별 최고 조합만 (분류) ===")
print(best_cls.to_string(index=False))

'''
=== 전체 조합 결과 (분류) ===
       dataset task                              pipeline  accuracy  f1_macro
        cancer 이진분류                    Standard -> LogReg   1.00000   1.00000
        cancer 이진분류                         MinMax -> SVC   0.99123   0.99062
        cancer 이진분류                      MinMax -> LogReg   0.99123   0.99053
        cancer 이진분류         MinMax -> PCA(0.95) -> LogReg   0.99123   0.99053
        cancer 이진분류       Standard -> PCA(0.95) -> LogReg   0.99123   0.99062
        cancer 이진분류                          RandomForest   0.98246   0.98096
        cancer 이진분류                MinMax -> RandomForest   0.98246   0.98096
        cancer 이진분류   MinMax -> PCA(0.95) -> RandomForest   0.98246   0.98115
        cancer 이진분류            MinMax -> PCA(0.95) -> SVC   0.98246   0.98133
        cancer 이진분류              Standard -> RandomForest   0.98246   0.98096
        cancer 이진분류                       Standard -> SVC   0.98246   0.98133
        cancer 이진분류 Standard -> PCA(0.95) -> RandomForest   0.98246   0.98115
        cancer 이진분류          Standard -> PCA(0.95) -> SVC   0.97368   0.97186
        cancer 이진분류             PCA(0.95) -> RandomForest   0.86842   0.85791
dacon_diabetes 이진분류                    Standard -> LogReg   0.76336   0.72666
dacon_diabetes 이진분류       Standard -> PCA(0.95) -> LogReg   0.76336   0.72666
dacon_diabetes 이진분류                      MinMax -> LogReg   0.74809   0.70903
dacon_diabetes 이진분류         MinMax -> PCA(0.95) -> LogReg   0.74809   0.70903
dacon_diabetes 이진분류          Standard -> PCA(0.95) -> SVC   0.74046   0.69409
dacon_diabetes 이진분류                         MinMax -> SVC   0.73282   0.68727
dacon_diabetes 이진분류                       Standard -> SVC   0.73282   0.68727
dacon_diabetes 이진분류            MinMax -> PCA(0.95) -> SVC   0.72519   0.68049
dacon_diabetes 이진분류 Standard -> PCA(0.95) -> RandomForest   0.72519   0.70137
dacon_diabetes 이진분류                          RandomForest   0.70992   0.67490
dacon_diabetes 이진분류                MinMax -> RandomForest   0.70992   0.67490
dacon_diabetes 이진분류              Standard -> RandomForest   0.70992   0.67490
dacon_diabetes 이진분류   MinMax -> PCA(0.95) -> RandomForest   0.67939   0.64821
dacon_diabetes 이진분류             PCA(0.95) -> RandomForest   0.64885   0.61841
        digits 다중분류            MinMax -> PCA(0.95) -> SVC   0.99167   0.99166
        digits 다중분류                         MinMax -> SVC   0.98056   0.98051
        digits 다중분류                       Standard -> SVC   0.98056   0.98031
        digits 다중분류          Standard -> PCA(0.95) -> SVC   0.98056   0.98043
        digits 다중분류                    Standard -> LogReg   0.96667   0.96664
        digits 다중분류                          RandomForest   0.96389   0.96389
        digits 다중분류                MinMax -> RandomForest   0.96389   0.96389
        digits 다중분류              Standard -> RandomForest   0.96389   0.96389
        digits 다중분류       Standard -> PCA(0.95) -> LogReg   0.96389   0.96378
        digits 다중분류 Standard -> PCA(0.95) -> RandomForest   0.96111   0.96087
        digits 다중분류                      MinMax -> LogReg   0.95833   0.95837
        digits 다중분류   MinMax -> PCA(0.95) -> RandomForest   0.95556   0.95529
        digits 다중분류             PCA(0.95) -> RandomForest   0.95000   0.94933
        digits 다중분류         MinMax -> PCA(0.95) -> LogReg   0.95000   0.95017
   kaggle_bank 이진분류                                   SVC   0.86479   0.76212
   kaggle_bank 이진분류                      PCA(0.95) -> SVC   0.86479   0.76212
   kaggle_bank 이진분류             PCA(0.95) -> RandomForest   0.86054   0.76804
   kaggle_bank 이진분류                          RandomForest   0.86051   0.76897
   kaggle_bank 이진분류                                LogReg   0.82788   0.68162
   kaggle_bank 이진분류                   PCA(0.95) -> LogReg   0.82788   0.68162
          wine 다중분류                          RandomForest   1.00000   1.00000
          wine 다중분류                MinMax -> RandomForest   1.00000   1.00000
          wine 다중분류                         MinMax -> SVC   1.00000   1.00000
          wine 다중분류                      MinMax -> LogReg   1.00000   1.00000
          wine 다중분류            MinMax -> PCA(0.95) -> SVC   1.00000   1.00000
          wine 다중분류         MinMax -> PCA(0.95) -> LogReg   1.00000   1.00000
          wine 다중분류              Standard -> RandomForest   1.00000   1.00000
          wine 다중분류                       Standard -> SVC   1.00000   1.00000
          wine 다중분류 Standard -> PCA(0.95) -> RandomForest   1.00000   1.00000
          wine 다중분류          Standard -> PCA(0.95) -> SVC   1.00000   1.00000
          wine 다중분류                    Standard -> LogReg   0.97222   0.97178
          wine 다중분류       Standard -> PCA(0.95) -> LogReg   0.97222   0.97432
          wine 다중분류   MinMax -> PCA(0.95) -> RandomForest   0.94444   0.94582
          wine 다중분류             PCA(0.95) -> RandomForest   0.52778   0.50047

=== 데이터셋별 최고 조합만 (분류) ===
       dataset task                   pipeline  accuracy  f1_macro
        cancer 이진분류         Standard -> LogReg   1.00000   1.00000
dacon_diabetes 이진분류         Standard -> LogReg   0.76336   0.72666
        digits 다중분류 MinMax -> PCA(0.95) -> SVC   0.99167   0.99166
   kaggle_bank 이진분류                        SVC   0.86479   0.76212
          wine 다중분류               RandomForest   1.00000   1.00000
'''