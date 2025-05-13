# 02. california
# 03. diabetes
# 04. dacon_ddarung
# 05. kaggle_bike
# m64_2 와 동일하게 회귀 모델 만들기

# [결과 예:]
# 캘리포니아, 민맥스, 1.0
# 다이아비티스, 스텐다드, 0.99

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

RANDOM_STATE = 333
TEST_SIZE = 0.2

# ─────────────────────────────────────────
# 1) 데이터 로더들 (각각 X, y, pre_scaled 반환)
# ─────────────────────────────────────────
def _load_california():
    d = fetch_california_housing()
    return d.data, d.target, False

def _load_diabetes_reg():
    d = load_diabetes()
    return d.data, d.target, False

def _load_dacon_ddarung():
    path = 'C:\Study25\_data\dacon\따릉이\\'
    train_csv = pd.read_csv(path + 'train.csv', index_col=0)
    test_csv  = pd.read_csv(path + 'test.csv',  index_col=0)
    _sub      = pd.read_csv(path + 'submission.csv', index_col=0)  # 사용 안 함

    train_csv = train_csv.dropna()
    test_csv  = test_csv.fillna(test_csv.mean())

    X = train_csv.drop(['count'], axis=1)
    y = train_csv['count']
    return X.values, y.values, False

def _load_kaggle_bike():
    path = r'C:\Study25\_data\kaggle\bike\\'
    train_csv    = pd.read_csv(path + 'train.csv', index_col=0)
    _new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)  # 사용 안 함
    _submission   = pd.read_csv(path + 'sampleSubmission.csv')      # 사용 안 함

    X = train_csv.drop(['count'], axis=1)
    y = train_csv['count']
    return X.values, y.values, False

# 매핑
LOADERS_REG = {
    "california": _load_california,
    "diabetes": _load_diabetes_reg,
    "dacon_ddarung": _load_dacon_ddarung,
    "kaggle_bike": _load_kaggle_bike,
}

def load_dataset_reg(name: str):
    name = name.lower()
    if name not in LOADERS_REG:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS_REG)}")
    return LOADERS_REG[name]()

# ─────────────────────────────────────────
# 2) 스케일러 / 모델
# ─────────────────────────────────────────
scalers = {
    "None": None,
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler(),
}

models = {
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
    "GBDT": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
    "SVR": SVR(),  # X 스케일에 민감
}

datasets = [
    ("california", "회귀"),
    ("diabetes", "회귀"),
    ("dacon_ddarung", "회귀"),
    ("kaggle_bike", "회귀"),
]

# ─────────────────────────────────────────
# 3) 실행
# ─────────────────────────────────────────
rows = []

for ds_name, task in datasets:
    X, y, pre_scaled = load_dataset_reg(ds_name)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    for scaler_name, scaler in scalers.items():
        # 현재 회귀 로더들은 모두 pre_scaled=False라서 아래 조건은 항상 False지만,
        # 패턴 통일을 위해 남겨둠
        if pre_scaled and scaler is not None:
            continue

        for model_name, model in models.items():
            if scaler is None:
                estimator = model
                pipeline_desc = model_name
            else:
                estimator = make_pipeline(scaler, model)
                pipeline_desc = f"{scaler_name} -> {model_name}"

            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            rows.append({
                "dataset": ds_name,
                "pipeline": pipeline_desc,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            })

df = pd.DataFrame(rows)
df["R2"]   = df["R2"].round(5)
df["RMSE"] = df["RMSE"].round(4)
df["MAE"]  = df["MAE"].round(4)

print("=== 전체 조합 결과 (회귀) ===")
print(df.sort_values(by=["dataset", "R2"], ascending=[True, False]).to_string(index=False))

# 최고 조합 (R² 내림차순 → RMSE 오름차순 → MAE 오름차순)
best_reg = df.sort_values(
    by=["dataset", "R2", "RMSE", "MAE"],
    ascending=[True, False, True, True]
).groupby("dataset", as_index=False).head(1).reset_index(drop=True)

print("\n=== 데이터셋별 최고 조합만 (회귀) ===")
print(best_reg.to_string(index=False))

# (선택) 저장
# df.to_csv("reg_all_results.csv", index=False)
# best_reg.to_csv("reg_best_per_dataset.csv", index=False)

'''
=== 전체 조합 결과 (회귀) ===
      dataset                 pipeline       R2    RMSE     MAE
   california   MinMax -> RandomForest  0.79905  0.5060  0.3295
   california Standard -> RandomForest  0.79896  0.5061  0.3294
   california   Robust -> RandomForest  0.79887  0.5062  0.3295
   california             RandomForest  0.79871  0.5064  0.3296
   california                     GBDT  0.78048  0.5288  0.3656
   california           MinMax -> GBDT  0.78045  0.5289  0.3656
   california         Standard -> GBDT  0.78045  0.5289  0.3656
   california           Robust -> GBDT  0.78045  0.5289  0.3656
   california          Standard -> SVR  0.73241  0.5839  0.3907
   california            Robust -> SVR  0.67494  0.6435  0.4493
   california            MinMax -> SVR  0.66351  0.6548  0.4440
   california        Standard -> Ridge  0.59983  0.7140  0.5252
   california                    Ridge  0.59982  0.7140  0.5252
   california          Robust -> Ridge  0.59981  0.7141  0.5252
   california          MinMax -> Ridge  0.59592  0.7175  0.5332
   california                    Lasso  0.29318  0.9490  0.7434
   california          MinMax -> Lasso -0.00099  1.1293  0.8883
   california        Standard -> Lasso -0.00099  1.1293  0.8883
   california          Robust -> Lasso -0.00099  1.1293  0.8883
   california                      SVR -0.01164  1.1353  0.8384
dacon_ddarung           MinMax -> GBDT  0.79061 37.5525 26.3514
dacon_ddarung           Robust -> GBDT  0.79047 37.5652 26.3719
dacon_ddarung         Standard -> GBDT  0.78978 37.6269 26.4155
dacon_ddarung                     GBDT  0.78956 37.6461 26.4293
dacon_ddarung   Robust -> RandomForest  0.78002 38.4898 25.7686
dacon_ddarung Standard -> RandomForest  0.77897 38.5820 25.8097
dacon_ddarung   MinMax -> RandomForest  0.77868 38.6072 25.7797
dacon_ddarung             RandomForest  0.77809 38.6584 25.7665
dacon_ddarung          MinMax -> Ridge  0.60187 51.7810 38.9559
dacon_ddarung                    Ridge  0.60176 51.7878 39.0050
dacon_ddarung        Standard -> Ridge  0.60130 51.8181 39.0278
dacon_ddarung          Robust -> Ridge  0.60121 51.8240 39.0171
dacon_ddarung        Standard -> Lasso  0.60093 51.8423 38.9975
dacon_ddarung                    Lasso  0.59213 52.4103 39.2051
dacon_ddarung          Robust -> Lasso  0.59130 52.4636 39.2797
dacon_ddarung          MinMax -> Lasso  0.58397 52.9322 39.2262
dacon_ddarung            MinMax -> SVR  0.51181 57.3391 40.1312
dacon_ddarung          Standard -> SVR  0.48421 58.9379 42.2727
dacon_ddarung            Robust -> SVR  0.46465 60.0450 43.6544
dacon_ddarung                      SVR  0.09153 78.2192 60.6505
     diabetes          MinMax -> Ridge  0.46457 53.2802 42.2097
     diabetes                    Ridge  0.46357 53.3299 44.2856
     diabetes          Robust -> Lasso  0.46144 53.4357 42.4518
     diabetes          MinMax -> Lasso  0.46044 53.4857 43.5774
     diabetes        Standard -> Lasso  0.45763 53.6244 42.5468
     diabetes        Standard -> Ridge  0.45688 53.6617 42.3681
     diabetes          Robust -> Ridge  0.45617 53.6968 42.4198
     diabetes Standard -> RandomForest  0.38626 57.0439 47.0827
     diabetes             RandomForest  0.38566 57.0717 47.1319
     diabetes   MinMax -> RandomForest  0.38557 57.0757 47.0216
     diabetes   Robust -> RandomForest  0.38461 57.1206 47.1337
     diabetes                    Lasso  0.34941 58.7314 51.1057
     diabetes           Robust -> GBDT  0.32548 59.8016 50.0704
     diabetes           MinMax -> GBDT  0.32072 60.0125 50.2889
     diabetes         Standard -> GBDT  0.32008 60.0405 50.3184
     diabetes                     GBDT  0.31613 60.2149 50.4587
     diabetes            Robust -> SVR  0.17183 66.2639 56.5070
     diabetes          Standard -> SVR  0.16409 66.5725 56.8278
     diabetes                      SVR  0.16366 66.5898 56.8598
     diabetes            MinMax -> SVR  0.14438 67.3530 57.8335
  kaggle_bike                    Ridge  1.00000  0.0000  0.0000
  kaggle_bike                    Lasso  1.00000  0.0224  0.0149
  kaggle_bike        Standard -> Ridge  1.00000  0.0188  0.0138
  kaggle_bike          Robust -> Ridge  1.00000  0.0293  0.0210
  kaggle_bike          MinMax -> Ridge  0.99999  0.6546  0.4947
  kaggle_bike        Standard -> Lasso  0.99996  1.1523  0.9071
  kaggle_bike          Robust -> Lasso  0.99995  1.2386  0.9876
  kaggle_bike   Robust -> RandomForest  0.99971  3.0254  1.1282
  kaggle_bike             RandomForest  0.99970  3.0405  1.1317
  kaggle_bike   MinMax -> RandomForest  0.99970  3.0399  1.1254
  kaggle_bike Standard -> RandomForest  0.99970  3.0461  1.1350
  kaggle_bike                     GBDT  0.99908  5.3541  3.3933
  kaggle_bike           MinMax -> GBDT  0.99908  5.3521  3.3933
  kaggle_bike         Standard -> GBDT  0.99908  5.3685  3.4057
  kaggle_bike           Robust -> GBDT  0.99908  5.3557  3.3964
  kaggle_bike          MinMax -> Lasso  0.99804  7.8189  5.9162
  kaggle_bike                      SVR  0.96716 32.0341  7.5464
  kaggle_bike            MinMax -> SVR  0.92758 47.5758 28.5636
  kaggle_bike            Robust -> SVR  0.88531 59.8694 25.5184
  kaggle_bike          Standard -> SVR  0.86415 65.1580 30.6631

=== 데이터셋별 최고 조합만 (회귀) ===
      dataset               pipeline      R2    RMSE     MAE
   california MinMax -> RandomForest 0.79905  0.5060  0.3295
dacon_ddarung         MinMax -> GBDT 0.79061 37.5525 26.3514
     diabetes        MinMax -> Ridge 0.46457 53.2802 42.2097
  kaggle_bike                  Ridge 1.00000  0.0000  0.0000
'''