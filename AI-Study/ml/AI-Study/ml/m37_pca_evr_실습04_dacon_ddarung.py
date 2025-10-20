# pca로 데이터를 축소했을 때 성능이 좋아지는지를 찾는다.
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


path = 'C:\Study25\_data\dacon\따릉이\\'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)             # [715 rows x 1 columns]

print(train_csv.shape)            #(1459, 10)
print(test_csv.shape)             #(715, 9)
print(submission_csv.shape)       #(715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')           # Nan = 결측치              이상치 ex. 서울의 온도 41도

print(train_csv.info())           # 결측치 확인

print(train_csv.describe())

########################   결측치 처리 1. 삭제   ######################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum())     # 위 함수와 똑같음

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [1328 rows x 10 columns]

########################   결측치 처리 2. 평균값 넣기   ######################
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum()) 
# print(train_csv.info()) 

########################   테스트 데이터의 결측치 확인   ######################
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null

#====================== x 와 y 데이터를 나눠준다 =========================#
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
                #  count라는 axis=1 열 삭제, 참고로 행 삭제는 axis=0
print(x)                                 # [1459 rows x 9 columns]
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다
print(y.shape)                           #(1459,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (1062, 9) (266, 9)

# 데이터 모아서 한번에 때려보자
x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  # (1328, 9)

# exit()
pca = PCA(n_components=9) # 열 최대값
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# x = x.reshape(70000, 28*28) # (70000, 784)

thresholds = [0.95, 0.99, 0.999, 1.0]

for threshold in thresholds:
    n_components = np.argmax(evr_cumsum >= threshold) + 1
    print(f"{threshold:.3f} 이상을 만족하는 주성분 개수: {n_components}개")

# 0.950 이상을 만족하는 주성분 개수: 8개
# 0.990 이상을 만족하는 주성분 개수: 9개
# 0.999 이상을 만족하는 주성분 개수: 9개
# 1.000 이상을 만족하는 주성분 개수: 1개


# exit()
# 2. PCA 차원 수 리스트
num_components_list = [8, 9, 1]


# 3. DNN 모델 정의 함수
def build_dnn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam',)
    return model

# 4. 반복 실행
for n in num_components_list:
    print(f"\n▶ n_components = {n}")
    
    # PCA
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x)
    
    # train/test split
    x_train_pca, x_test_pca, y_train_split, y_test_split = train_test_split(
        x_pca, y, test_size=0.2, random_state=42, #stratify=np.argmax(y, axis=1)
    )
    
    # DNN 모델 생성 및 학습
    model = build_dnn(n)
    model.fit(x_train_pca, y_train_split, epochs=10, batch_size=128,
              validation_split=0.2, verbose=0)
    
    # 평가
    loss = model.evaluate(x_test_pca, y_test_split, verbose=0)
    # 예측
    y_pred = model.predict(x_test_pca, verbose=0)
    
    # 성능 평가
    r2 = r2_score(y_test_split, y_pred)
    print(f"R2 Score: {r2:.4f}")
    
# ▶ n_components = 8
# R2 Score: -1.1929

# ▶ n_components = 9
# R2 Score: -1.2849

# ▶ n_components = 1
# R2 Score: -1.3406