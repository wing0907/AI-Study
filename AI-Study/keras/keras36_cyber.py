import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

# 1. Load Data
path = 'C:/Study25/_data/dacon/cyber_attack\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Split x and y
# x = train_csv.drop(['attack_type'], axis=1)
# y = train_csv['attack_type']

print(train_csv.shape)            #(11999, 21)
print(test_csv.shape)             #(3000, 20)
print(submission_csv.shape)       #(3000, 1)

print(train_csv.columns)
# Index(['ip_src', 'port_src', 'ip_dst', 'port_dst', 'protocol', 'duration',  
#        'pkt_count_fwd', 'pkt_count_bwd', 'rate_fwd_pkts', 'rate_bwd_pkts',  
#        'rate_fwd_bytes', 'rate_bwd_bytes', 'payload_fwd_mean',
#        'payload_bwd_mean', 'tcp_win_fwd_init', 'tcp_win_bwd_init',
#        'tcp_syn_count', 'tcp_psh_count', 'tcp_rst_count', 'iat_avg_packets',
#        'attack_type'],
#       dtype='object')
print(train_csv.info())         # 결측치 확인

print(train_csv.describe())
print(train_csv.isnull().sum())

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
test_csv = test_csv.fillna(0)

print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [2532 rows x 21 columns]
print(test_csv.info())         # 결측치 확인
print(test_csv)                 # [613 rows x 20 columns]

# ip를 0또는 1로 만들거나 drop 하기.
label_cols = ['protocol', 'attack_type']  # 라벨 인코딩 할 열 입력
label_encoders = {}                       # column 별로 라벨링을 다르게 해야함. 0과1로 훈련을 해서 문자값으로 받고 다시 inverse 해야 함.
for col in label_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    label_encoders[col] = le
    if col =='protocol' :
        test_csv[col]= le.transform(test_csv[col])


# 4. 최종 분리
x = train_csv.drop(['attack_type','ip_src', 'ip_dst'], axis=1)
y = train_csv['attack_type']

print(y.isnull().sum())  # 0이어야 정상
print(x.shape, y.shape)  # (2532, 18) (2532,)

from sklearn.preprocessing import OneHotEncoder
# y = y.values
# y = y.reshape(-1, 1)                # reshape 바뀌지 않아야 할것 2가지 =  1. 안에 들어간 값  / 2. 순서

# ohe = OneHotEncoder()               # metrics형태를 받기때문에 (N, 1)로 reshape하고 해야한다.
# y = ohe.fit_transform(y)#.toarray()  # .toarray()를 쓰면 numpy 형태로 바뀜
# print(y)                            # 희소행렬방식
# print(y.shape)   # (2532, 12)
# print(type(y))                      # <class 'scipy.sparse._csr.csr_matrix'>
    

y = pd.get_dummies(y)
print(y)            # [2532 rows x 12 columns]
print(y.shape)      # (2532, 12)

test_csv1 = test_csv.drop(['ip_src', 'ip_dst'], axis=1)
print(test_csv) # [3000 rows x 20 columns]


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv1)


# 6. Split train/test
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=19
)

# 7. Compute class weights
# weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights = dict(enumerate(weights))

# 8. Model
model = Sequential()
model.add(Dense(128, input_dim=18, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(12, activation='softmax'))

# 9. Compile
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 10. Callback
es = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=50,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
print(date)    
print(type(date))  
date = date.strftime("%m%d_%H%M")     
print(date)    
print(type(date))   

path_save = './_data/dacon/cyber_attack/'
filepath = path_save + f'k36_{date}_' + '{epoch:04d}-{val_loss:.4f}.hdf5'



mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)


# 11. Train
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=64,
    callbacks=[es, mcp],
    # class_weight=class_weights,
    verbose=1
)

# 12. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', round(loss, 4), 'Accuracy:', round(acc, 4))

# 13. Predict on test split
y_pred_prob = model.predict(x_test)
y_pred_binary = (y_pred_prob > 0.45).astype(int)
acc_score = accuracy_score(y_test, y_pred_binary)
print('Final Accuracy Score:', round(acc_score, 4))

# 14. Predict on submission set
y_submit = model.predict(test_scaled)
y_submit = np.argmax(y_submit, axis=1)



y_submit = label_encoders['attack_type'].inverse_transform(y_submit)

submission_csv['attack_type'] = y_submit
submission_csv.to_csv(path + f'submission_{date}_cyber_attack.csv')
print("Submission saved!")

# Evaluation - Loss: 0.1547 Accuracy: 0.9507
# Final Accuracy Score: 0.9448
