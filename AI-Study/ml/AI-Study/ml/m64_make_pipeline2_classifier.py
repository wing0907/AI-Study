# 이진분류
# 06. cancer
# 07. dacon_당뇨병
# 08. kaggle_bank

# 다중분류
# 09. wine
# 11. digits

##################
# 1. 데이터셋
# 2. 스케일러
# 3. 모델

# [결과] 6개의 데이터셋에서 어떤 스케일러와
# 어떤 모델을 썻을때 성능이 얼마인지
# 출력시키기

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RANDOM_STATE = 333
TEST_SIZE = 0.2

# ---------------------------
# 1) 내장 데이터셋 로더
# ---------------------------
def load_dataset(name):
    name = name.lower()
    if name in ["cancer", "breast_cancer"]:
        x, y = load_breast_cancer(return_X_y=True)
    elif name == "wine":
        x, y = load_wine(return_X_y=True)
    elif name == "digits":
        x, y = load_digits(return_X_y=True)
    elif name == "dacon_당뇨병":
        x, y = load_dacon_diabetes()
    elif name == "kaggle_bank":
        x, y = load_kaggle_bank()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return x, y

# ---------------------------
# 2) dacon 당뇨병 전처리
# ---------------------------
def load_dacon_diabetes():
    path = r'C:\Study25\_data\dacon\diabetes\\'
    train_csv = pd.read_csv(path + 'train.csv', index_col=0)

    X = train_csv.drop('Outcome', axis=1)
    y = train_csv['Outcome']

    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[zero_not_allowed] = X[zero_not_allowed].replace(0, np.nan)
    X = X.fillna(X.mean())

    return X.values, y.values

# ---------------------------
# 3) kaggle_bank 전처리
# ---------------------------
def load_kaggle_bank():
    path = 'C:\Study25\_data\kaggle\\bank\\'
    train_csv = pd.read_csv(path + 'train.csv', index_col=0)

    from sklearn.preprocessing import LabelEncoder
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()

    train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
    train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

    train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)

    x = train_csv.drop(['Exited'], axis=1)
    y = train_csv['Exited']

    # EstimatedSalary만 별도 스케일링
    x_other = x.drop(['EstimatedSalary'], axis=1)
    x_salary = x[['EstimatedSalary']]

    scaler_other = StandardScaler()
    scaler_salary = StandardScaler()

    x_other_scaled = scaler_other.fit_transform(x_other)
    x_salary_scaled = scaler_salary.fit_transform(x_salary)

    x_scaled = np.concatenate([x_other_scaled, x_salary_scaled], axis=1)

    return x_scaled, y.values

# ---------------------------
# 4) 스케일러 / 모델 목록
# ---------------------------
scalers = {
    "None": None,
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
}

models = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "SVC": SVC(),
}

datasets = [
    ("cancer", "이진분류"),
    ("dacon_당뇨병", "이진분류"),
    ("kaggle_bank", "이진분류"),
    ("wine", "다중분류"),
    ("digits", "다중분류"),
]

# ---------------------------
# 5) 실행 루프
# ---------------------------
results = []
for ds_name, task_type in datasets:
    X, Y = load_dataset(ds_name)
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y
    )

    for scaler_name, scaler_obj in scalers.items():
        for model_name, model_obj in models.items():
            if scaler_obj is None:
                model = model_obj
                pipeline_desc = model_name
            else:
                model = make_pipeline(scaler_obj, model_obj)
                pipeline_desc = f"{scaler_name} -> {model_name}"

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)

            results.append({
                "dataset": ds_name,
                "task": task_type,
                "pipeline": pipeline_desc,
                "accuracy": acc
            })

# ---------------------------
# 6) 결과 정리
# ---------------------------
df = pd.DataFrame(results).sort_values(
    by=["dataset", "accuracy"], ascending=[True, False]
).reset_index(drop=True)

print(df.to_string(index=False))


#   dataset task                       pipeline  accuracy
#      cancer 이진분류            MinMaxScaler -> SVC  0.991228
#      cancer 이진분류                   RandomForest  0.982456
#      cancer 이진분류   MinMaxScaler -> RandomForest  0.982456
#      cancer 이진분류 StandardScaler -> RandomForest  0.982456
#      cancer 이진분류          StandardScaler -> SVC  0.982456
#      cancer 이진분류                            SVC  0.938596
#   dacon_당뇨병 이진분류            MinMaxScaler -> SVC  0.732824
#   dacon_당뇨병 이진분류          StandardScaler -> SVC  0.732824
#   dacon_당뇨병 이진분류                            SVC  0.725191
#   dacon_당뇨병 이진분류                   RandomForest  0.709924
#   dacon_당뇨병 이진분류   MinMaxScaler -> RandomForest  0.709924
#   dacon_당뇨병 이진분류 StandardScaler -> RandomForest  0.709924
#      digits 다중분류                            SVC  0.980556
#      digits 다중분류            MinMaxScaler -> SVC  0.980556
#      digits 다중분류          StandardScaler -> SVC  0.980556
#      digits 다중분류                   RandomForest  0.963889
#      digits 다중분류   MinMaxScaler -> RandomForest  0.963889
#      digits 다중분류 StandardScaler -> RandomForest  0.963889
# kaggle_bank 이진분류          StandardScaler -> SVC  0.864817
# kaggle_bank 이진분류                            SVC  0.864786
# kaggle_bank 이진분류   MinMaxScaler -> RandomForest  0.861757
# kaggle_bank 이진분류 StandardScaler -> RandomForest  0.860666
# kaggle_bank 이진분류                   RandomForest  0.860514
# kaggle_bank 이진분류            MinMaxScaler -> SVC  0.859787
#        wine 다중분류                   RandomForest  1.000000
#        wine 다중분류   MinMaxScaler -> RandomForest  1.000000
#        wine 다중분류            MinMaxScaler -> SVC  1.000000
#        wine 다중분류 StandardScaler -> RandomForest  1.000000
#        wine 다중분류          StandardScaler -> SVC  1.000000
#        wine 다중분류                            SVC  0.638889
