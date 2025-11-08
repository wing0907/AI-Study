import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 데이터 불러오기
train = pd.read_csv("C:/Study25/_data/dacon/갑상선암/train.csv", index_col=0)
test = pd.read_csv("C:/Study25/_data/dacon/갑상선암/test.csv", index_col=0)

# 파생변수 생성
train['Radiation_Nodule'] = train['Radiation_History'].astype(str) + '_' + pd.cut(train['Nodule_Size'], bins=3, labels=["low", "mid", "high"]).astype(str)
test['Radiation_Nodule'] = test['Radiation_History'].astype(str) + '_' + pd.cut(test['Nodule_Size'], bins=3, labels=["low", "mid", "high"]).astype(str)
train['Is_Senior'] = (train['Age'] > 60).astype(int)
test['Is_Senior'] = (test['Age'] > 60).astype(int)

# 인코딩
categorical = train.select_dtypes(include='object').columns
all_data = pd.concat([train.drop('Cancer', axis=1), test], axis=0)
all_data = pd.get_dummies(all_data, columns=categorical, drop_first=True)
x = all_data.iloc[:len(train), :]
y = train['Cancer']

# 결과 저장 리스트
results = []
thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

for thresh in thresholds:
    xgb_temp = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_temp.fit(x, y)
    importances = xgb_temp.feature_importances_
    selected = x.columns[importances > thresh]

    x_sel = x[selected]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_sel)

    x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    f1 = f1_score(y_val, pred)

    results.append((thresh, len(selected), f1))

# 결과 출력
df_result = pd.DataFrame(results, columns=['Threshold', 'Num_Features', 'F1_Score'])
print(df_result)


#    Threshold  Num_Features  F1_Score
# 0       0.01            11  0.370459
# 1       0.02             8  0.423976
# 2       0.03             8  0.423976
# 3       0.04             8  0.423976
# 4       0.05             5  0.391811