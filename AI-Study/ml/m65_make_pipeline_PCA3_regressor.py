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
from sklearn.decomposition import PCA

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
# 2) 스케일러 / PCA / 모델
# ─────────────────────────────────────────
scalers = {
    "None": None,
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler(),
}

pca_options = {
    "NoPCA": None,
    "PCA(0.95)": PCA(n_components=0.95, random_state=RANDOM_STATE),
}

models = {
    "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
    "GBDT": GradientBoostingRegressor(random_state=RANDOM_STATE),
    # ↓↓↓ cholesky 경로 회피 (SciPy 의존도 낮은 solver)
    "Ridge": Ridge(solver="svd"),      # 또는 Ridge(solver="lsqr")
    "Lasso": Lasso(max_iter=10000),
    "SVR": SVR(),
}


# 스케일 필요한 회귀 모델들
NEEDS_SCALING = {"Ridge", "Lasso", "SVR"}

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
        # (모든 회귀 로더는 pre_scaled=False라 생략 조건은 없음. 패턴 유지 겸 남김)
        if pre_scaled and scaler is not None:
            continue

        for pca_name, pca_obj in pca_options.items():
            for model_name, model in models.items():
                # 스케일 필요한 모델인데 스케일러가 없으면 건너뜀
                if (model_name in NEEDS_SCALING) and (scaler is None):
                    continue

                steps = []
                if scaler is not None:
                    steps.append(scaler)
                if pca_obj is not None:
                    steps.append(pca_obj)
                steps.append(model)

                estimator = make_pipeline(*steps) if steps else model

                parts = []
                if scaler is not None:
                    parts.append(scaler_name)
                if pca_obj is not None:
                    parts.append(pca_name)
                parts.append(model_name)
                pipeline_desc = " -> ".join(parts) if parts else model_name

                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(x_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 버전 호환
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

# 최고 조합: R² ↓, RMSE ↑(작을수록 좋음), MAE ↑(작을수록 좋음)
best_reg = df.sort_values(
    by=["dataset", "R2", "RMSE", "MAE"],
    ascending=[True, False, True, True]
).groupby("dataset", as_index=False).head(1).reset_index(drop=True)

print("\n=== 데이터셋별 최고 조합만 (회귀) ===")
print(best_reg.to_string(index=False))

'''
=== 전체 조합 결과 (회귀) ===
      dataset                              pipeline       R2    RMSE     MAE
   california              Standard -> RandomForest  0.79955  0.5054  0.3297
   california                MinMax -> RandomForest  0.79953  0.5054  0.3293
   california                Robust -> RandomForest  0.79939  0.5056  0.3298
   california                          RandomForest  0.79926  0.5057  0.3299
   california                                  GBDT  0.78048  0.5288  0.3656
   california                      Standard -> GBDT  0.78045  0.5289  0.3656
   california                        Robust -> GBDT  0.78045  0.5289  0.3656
   california                        MinMax -> GBDT  0.77984  0.5296  0.3667
   california                       Standard -> SVR  0.73241  0.5839  0.3907
   california                         Robust -> SVR  0.67494  0.6435  0.4493
   california                         MinMax -> SVR  0.66351  0.6548  0.4440
   california Standard -> PCA(0.95) -> RandomForest  0.65500  0.6630  0.4699
   california   MinMax -> PCA(0.95) -> RandomForest  0.64013  0.6771  0.4754
   california          Standard -> PCA(0.95) -> SVR  0.63938  0.6778  0.4644
   california           MinMax -> PCA(0.95) -> GBDT  0.62773  0.6887  0.4988
   california         Standard -> PCA(0.95) -> GBDT  0.60965  0.7052  0.5106
   california                     Standard -> Ridge  0.59983  0.7140  0.5252
   california                       Robust -> Ridge  0.59981  0.7141  0.5252
   california                       MinMax -> Ridge  0.59592  0.7175  0.5332
   california            MinMax -> PCA(0.95) -> SVR  0.58991  0.7228  0.4951
   california          MinMax -> PCA(0.95) -> Ridge  0.53140  0.7727  0.5735
   california        Standard -> PCA(0.95) -> Ridge  0.50019  0.7980  0.5975
   california           Robust -> PCA(0.95) -> GBDT  0.08882  1.0775  0.8361
   california            Robust -> PCA(0.95) -> SVR  0.06320  1.0925  0.8168
   california                       MinMax -> Lasso -0.00099  1.1293  0.8883
   california          MinMax -> PCA(0.95) -> Lasso -0.00099  1.1293  0.8883
   california                     Standard -> Lasso -0.00099  1.1293  0.8883
   california        Standard -> PCA(0.95) -> Lasso -0.00099  1.1293  0.8883
   california                       Robust -> Lasso -0.00099  1.1293  0.8883
   california          Robust -> PCA(0.95) -> Lasso -0.00099  1.1293  0.8883
   california                     PCA(0.95) -> GBDT -0.00246  1.1301  0.8877
   california          Robust -> PCA(0.95) -> Ridge -0.00331  1.1306  0.8876
   california   Robust -> PCA(0.95) -> RandomForest -0.05731  1.1606  0.9015
   california             PCA(0.95) -> RandomForest -0.52618  1.3944  1.0852
dacon_ddarung                        MinMax -> GBDT  0.79061 37.5525 26.3514
dacon_ddarung                        Robust -> GBDT  0.79047 37.5652 26.3719
dacon_ddarung                      Standard -> GBDT  0.78978 37.6269 26.4155
dacon_ddarung                                  GBDT  0.78956 37.6461 26.4293
dacon_ddarung                Robust -> RandomForest  0.78002 38.4898 25.7686
dacon_ddarung              Standard -> RandomForest  0.77897 38.5820 25.8097
dacon_ddarung                MinMax -> RandomForest  0.77868 38.6072 25.7797
dacon_ddarung                          RandomForest  0.77809 38.6584 25.7665
dacon_ddarung   MinMax -> PCA(0.95) -> RandomForest  0.75900 40.2875 30.0562
dacon_ddarung Standard -> PCA(0.95) -> RandomForest  0.73870 41.9494 31.3018
dacon_ddarung   Robust -> PCA(0.95) -> RandomForest  0.72461 43.0660 32.2941
dacon_ddarung         Standard -> PCA(0.95) -> GBDT  0.70176 44.8170 33.6579
dacon_ddarung           MinMax -> PCA(0.95) -> GBDT  0.70156 44.8318 32.9565
dacon_ddarung           Robust -> PCA(0.95) -> GBDT  0.69005 45.6883 34.2240
dacon_ddarung                       MinMax -> Ridge  0.60187 51.7810 38.9559
dacon_ddarung                     Standard -> Ridge  0.60130 51.8181 39.0278
dacon_ddarung        Standard -> PCA(0.95) -> Ridge  0.60127 51.8197 39.0313
dacon_ddarung                       Robust -> Ridge  0.60121 51.8240 39.0171
dacon_ddarung                     Standard -> Lasso  0.60093 51.8423 38.9975
dacon_ddarung        Standard -> PCA(0.95) -> Lasso  0.59889 51.9744 39.2121
dacon_ddarung          MinMax -> PCA(0.95) -> Ridge  0.59734 52.0747 39.1904
dacon_ddarung                       Robust -> Lasso  0.59130 52.4636 39.2797
dacon_ddarung          MinMax -> PCA(0.95) -> Lasso  0.58994 52.5512 39.2537
dacon_ddarung          Robust -> PCA(0.95) -> Ridge  0.58488 52.8740 39.3279
dacon_ddarung                       MinMax -> Lasso  0.58397 52.9322 39.2262
dacon_ddarung          Robust -> PCA(0.95) -> Lasso  0.58376 52.9457 39.4643
dacon_ddarung            MinMax -> PCA(0.95) -> SVR  0.51408 57.2058 40.1621
dacon_ddarung                         MinMax -> SVR  0.51181 57.3391 40.1312
dacon_ddarung          Standard -> PCA(0.95) -> SVR  0.48745 58.7525 42.1385
dacon_ddarung                       Standard -> SVR  0.48421 58.9379 42.2727
dacon_ddarung            Robust -> PCA(0.95) -> SVR  0.46501 60.0250 43.6000
dacon_ddarung                         Robust -> SVR  0.46465 60.0450 43.6544
dacon_ddarung                     PCA(0.95) -> GBDT  0.07394 78.9727 61.7696
dacon_ddarung             PCA(0.95) -> RandomForest -0.36130 95.7492 73.9515
     diabetes          MinMax -> PCA(0.95) -> Ridge  0.50543 51.2072 40.4352
     diabetes          MinMax -> PCA(0.95) -> Lasso  0.49512 51.7381 40.7433
     diabetes                       MinMax -> Ridge  0.46458 53.2799 42.2094
     diabetes          Robust -> PCA(0.95) -> Lasso  0.46279 53.3689 42.5378
     diabetes                       Robust -> Lasso  0.46145 53.4353 42.4515
     diabetes        Standard -> PCA(0.95) -> Lasso  0.46137 53.4393 42.3125
     diabetes                       MinMax -> Lasso  0.46044 53.4854 43.5770
     diabetes                     Standard -> Lasso  0.45764 53.6241 42.5466
     diabetes                     Standard -> Ridge  0.45689 53.6613 42.3680
     diabetes                       Robust -> Ridge  0.45618 53.6964 42.4197
     diabetes        Standard -> PCA(0.95) -> Ridge  0.44909 54.0453 42.7937
     diabetes          Robust -> PCA(0.95) -> Ridge  0.44595 54.1988 43.1802
     diabetes Standard -> PCA(0.95) -> RandomForest  0.40692 56.0755 46.8912
     diabetes   MinMax -> PCA(0.95) -> RandomForest  0.39846 56.4738 43.3373
     diabetes                     PCA(0.95) -> GBDT  0.38785 56.9697 47.0416
     diabetes                          RandomForest  0.38714 57.0027 47.0826
     diabetes              Standard -> RandomForest  0.38669 57.0238 47.0438
     diabetes         Standard -> PCA(0.95) -> GBDT  0.38662 57.0269 47.0619
     diabetes                MinMax -> RandomForest  0.38625 57.0441 46.9700
     diabetes                Robust -> RandomForest  0.38587 57.0621 47.1130
     diabetes             PCA(0.95) -> RandomForest  0.38065 57.3038 48.4371
     diabetes   Robust -> PCA(0.95) -> RandomForest  0.37392 57.6145 47.7836
     diabetes           MinMax -> PCA(0.95) -> GBDT  0.35090 58.6638 44.0775
     diabetes                        MinMax -> GBDT  0.33129 59.5436 49.7975
     diabetes                      Standard -> GBDT  0.33066 59.5718 49.8270
     diabetes                                  GBDT  0.32636 59.7627 50.0200
     diabetes                        Robust -> GBDT  0.32548 59.8016 50.0704
     diabetes           Robust -> PCA(0.95) -> GBDT  0.32290 59.9162 48.8207
     diabetes            Robust -> PCA(0.95) -> SVR  0.17791 66.0200 56.3543
     diabetes                         Robust -> SVR  0.17182 66.2640 56.5072
     diabetes          Standard -> PCA(0.95) -> SVR  0.16616 66.4902 56.7671
     diabetes                       Standard -> SVR  0.16409 66.5725 56.8278
     diabetes                         MinMax -> SVR  0.14438 67.3530 57.8334
     diabetes            MinMax -> PCA(0.95) -> SVR  0.14428 67.3569 57.7730
  kaggle_bike                     Standard -> Ridge  1.00000  0.0188  0.0138
  kaggle_bike                       Robust -> Ridge  1.00000  0.0293  0.0210
  kaggle_bike                       MinMax -> Ridge  0.99999  0.6546  0.4947
  kaggle_bike                     Standard -> Lasso  0.99996  1.1523  0.9071
  kaggle_bike                       Robust -> Lasso  0.99995  1.2386  0.9876
  kaggle_bike                Robust -> RandomForest  0.99971  3.0254  1.1282
  kaggle_bike                          RandomForest  0.99970  3.0405  1.1317
  kaggle_bike                MinMax -> RandomForest  0.99970  3.0399  1.1254
  kaggle_bike              Standard -> RandomForest  0.99970  3.0461  1.1350
  kaggle_bike             PCA(0.95) -> RandomForest  0.99956  3.6895  2.3304
  kaggle_bike                     PCA(0.95) -> GBDT  0.99910  5.3174  3.6468
  kaggle_bike                                  GBDT  0.99908  5.3541  3.3933
  kaggle_bike                        MinMax -> GBDT  0.99908  5.3521  3.3933
  kaggle_bike                      Standard -> GBDT  0.99908  5.3685  3.4057
  kaggle_bike                        Robust -> GBDT  0.99908  5.3557  3.3964
  kaggle_bike                       MinMax -> Lasso  0.99804  7.8189  5.9162
  kaggle_bike          Robust -> PCA(0.95) -> Ridge  0.99634 10.6923  8.8688
  kaggle_bike          Robust -> PCA(0.95) -> Lasso  0.99588 11.3455  9.4351
  kaggle_bike        Standard -> PCA(0.95) -> Ridge  0.99574 11.5448  8.8787
  kaggle_bike        Standard -> PCA(0.95) -> Lasso  0.99560 11.7318  9.1942
  kaggle_bike   Robust -> PCA(0.95) -> RandomForest  0.99185 15.9552 11.2415
  kaggle_bike          MinMax -> PCA(0.95) -> Ridge  0.98621 20.7615 16.2334
  kaggle_bike           Robust -> PCA(0.95) -> GBDT  0.98571 21.1341 15.7311
  kaggle_bike Standard -> PCA(0.95) -> RandomForest  0.98200 23.7182 16.6935
  kaggle_bike          MinMax -> PCA(0.95) -> Lasso  0.98148 24.0589 18.6449
  kaggle_bike   MinMax -> PCA(0.95) -> RandomForest  0.97969 25.1917 18.0524
  kaggle_bike         Standard -> PCA(0.95) -> GBDT  0.97210 29.5300 22.7033
  kaggle_bike           MinMax -> PCA(0.95) -> GBDT  0.96709 32.0727 24.9506
  kaggle_bike                         MinMax -> SVR  0.92758 47.5758 28.5636
  kaggle_bike                         Robust -> SVR  0.88531 59.8694 25.5184
  kaggle_bike            Robust -> PCA(0.95) -> SVR  0.87799 61.7499 27.5722
  kaggle_bike          Standard -> PCA(0.95) -> SVR  0.87175 63.3091 30.4347
  kaggle_bike                       Standard -> SVR  0.86415 65.1580 30.6631
  kaggle_bike            MinMax -> PCA(0.95) -> SVR  0.86179 65.7230 41.3906

=== 데이터셋별 최고 조합만 (회귀) ===
      dataset                     pipeline      R2    RMSE     MAE
   california     Standard -> RandomForest 0.79955  0.5054  0.3297
dacon_ddarung               MinMax -> GBDT 0.79061 37.5525 26.3514
     diabetes MinMax -> PCA(0.95) -> Ridge 0.50543 51.2072 40.4352
  kaggle_bike            Standard -> Ridge 1.00000  0.0188  0.0138
'''