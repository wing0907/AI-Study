from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터셋
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

data_names = ['iris', 'breast_cancer', 'digits', 'wine']

# 2. 모델 리스트
model_list = [
    LinearSVC(max_iter=10000),
    LogisticRegression(max_iter=5000),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# 3. 모델별 스케일링 필요 여부 리스트 (True면 스케일링 적용)
scale_required = [True, True, False, False]

# 4. 반복 실행
for d_idx, (x, y) in enumerate(data_list):
    print(f'\nDataset: {data_names[d_idx]} --------------------')

    for m_idx, model in enumerate(model_list):
        model_name = model.__class__.__name__

        # 조건에 따라 스케일링 적용
        if scale_required[m_idx]:
            scaler = StandardScaler()
            x_input = scaler.fit_transform(x)
        else:
            x_input = x

        model.fit(x_input, y)
        score = model.score(x_input, y)
        print(f'{model_name:<25} : {score:.4f}')

# Dataset: iris --------------------
# LinearSVC                 : 0.9467
# LogisticRegression        : 0.9733
# DecisionTreeClassifier    : 1.0000
# RandomForestClassifier    : 1.0000

# Dataset: breast_cancer -----------
# LinearSVC                 : 0.9877
# LogisticRegression        : 0.9877
# DecisionTreeClassifier    : 1.0000
# RandomForestClassifier    : 1.0000

# Dataset: digits ------------------
# LinearSVC                 : 0.9944
# LogisticRegression        : 0.9989
# DecisionTreeClassifier    : 1.0000
# RandomForestClassifier    : 1.0000

# Dataset: wine --------------------
# LinearSVC                 : 1.0000
# LogisticRegression        : 1.0000
# DecisionTreeClassifier    : 1.0000
# RandomForestClassifier    : 1.0000