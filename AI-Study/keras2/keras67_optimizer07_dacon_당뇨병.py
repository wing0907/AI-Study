import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 1. 데이터 불러오기
path = r'C:\Study25\_data\dacon\diabetes\\'
train_csv        = pd.read_csv(path + 'train.csv', index_col=0)
test_csv         = pd.read_csv(path + 'test.csv',  index_col=0)
submission_csv   = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. 특성과 라벨 분리
X = train_csv.drop('Outcome', axis=1)
y = train_csv['Outcome']

# 3. 0 값을 NaN 으로 바꾸기
zero_not_allowed = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
X[zero_not_allowed] = X[zero_not_allowed].replace(0, np.nan)

# 4. NaN을 각 컬럼 평균으로 채우기
X = X.fillna(X.mean())

# 5. 스케일링
scaler       = RobustScaler()
X_scaled     = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_csv)

# 6. 학습/검증 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=36,
    stratify=y
)

# 7. 하이퍼파라미터 탐색 (1 epoch 테스트)
optimizers = [Adam, Adagrad, SGD, RMSprop]
lr_list    = [0.1, 0.01, 0.05, 0.001, 0.0001]

best_acc   = 0.0
best_opt   = None
best_lr    = None

for Opt in optimizers:
    for lr in lr_list:
        # 모델 정의
        model = Sequential([
            Dense(400, input_dim=x_train.shape[1], activation='relu'),
            Dense(400, activation='relu'),
            Dense(300, activation='relu'),
            Dense(300, activation='relu'),
            Dense(300, activation='relu'),
            Dense(50,  activation='relu'),
            Dense(20,  activation='relu'),
            Dense(1,   activation='sigmoid')
        ])
        model.compile(
            optimizer=Opt(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        # 1 epoch 학습
        model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )
        # 검증 평가
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        print(f"{Opt.__name__:<7} lr={lr:<7} → val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_opt = Opt
            best_lr  = lr

print(f"\n▶ Best optimizer: {best_opt.__name__}, lr={best_lr}, val_acc={best_acc:.4f}")

# 8. 최적 하이퍼파라미터로 모델 재학습 (with callbacks)
model = Sequential([
    Dense(400, input_dim=X_scaled.shape[1], activation='relu'),
    Dense(400, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(50,  activation='relu'),
    Dense(20,  activation='relu'),
    Dense(1,   activation='sigmoid')
])
model.compile(
    optimizer=best_opt(learning_rate=best_lr),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
es  = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                    verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                      mode='max', save_best_only=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                        factor=0.5, patience=10, verbose=1)

start = time.time()
history = model.fit(
    X_scaled, y,
    epochs=100,
    batch_size=32,
    verbose=2,
    validation_split=0.1,
    callbacks=[es, mcp, rlr],
)
end = time.time()
print(f"▶ Training time: {end - start:.2f}s")

# 9. 최종 검증 정확도
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"Final Validation Accuracy: {val_acc:.4f}")


# # 10. 테스트 세트 예측 및 제출 파일 생성
# y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int).reshape(-1)
# submission_csv['Outcome'] = y_test_pred
# submission_csv.to_csv('submission.csv', index=True)
# print("▶ submission.csv 생성 완료")

# ▶ Best optimizer: Adam, lr=0.001, val_acc=0.7328
# Epoch 00021: early stopping
# ▶ Training time: 3.31s
# Final Validation Accuracy: 0.7328