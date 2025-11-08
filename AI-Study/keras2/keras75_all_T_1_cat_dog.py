from tensorflow.keras.applications import (VGG16, Xception,
ResNet101, InceptionV3, ResNet50, InceptionResNetV2,
DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0)

########## GAP 쓰기, 기존꺼 최고 거랑 성능 비교 ############

model_list = [
    VGG16(include_top=False, input_shape=(32, 32, 3)),
    ResNet50(include_top=False, input_shape=(32, 32, 3)),
    ResNet101(include_top=False, input_shape=(32, 32, 3)),
    DenseNet121(include_top=False, input_shape=(32, 32, 3)),
    # InceptionV3(include_top=False, input_shape=(32, 32, 3)),
    # InceptionResNetV2(include_top=False, input_shape=(32, 32, 3)),
    MobileNetV2(include_top=False, input_shape=(32, 32, 3)),
    # NASNetMobile(include_top=False, input_shape=(32, 32, 3)),
    EfficientNetB0(include_top=False, input_shape=(32, 32, 3)),
    # Xception(include_top=False, input_shape=(32, 32, 3)),
]
  
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# --- 1. Setup and Configuration ---
SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 1. 환경 설정 및 시드 고정 ---
# 재현 가능한 결과를 위해 시드를 고정합니다.
SEED = 333
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. 데이터 로드 및 전처리 ---
# 사용자의 로컬 NPY 파일을 불러옵니다.
# NOTE: 파일 경로를 사용자의 환경에 맞게 수정해야 합니다.
np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
try:
    print(f"'{np_path}keras44_01_x_train.npy' 파일을 불러오는 중...")
    x_train = np.load(np_path + "keras44_01_x_train.npy")
    print(f"'{np_path}keras44_01_y_train.npy' 파일을 불러오는 중...")
    y_train = np.load(np_path + "keras44_01_y_train.npy")
    print(f"'{np_path}keras44_01_x_test.npy' 파일을 불러오는 중...")
    x_test = np.load(np_path + "keras44_01_x_test.npy")
    print(f"'{np_path}keras44_01_y_test.npy' 파일을 불러오는 중...")
    y_test = np.load(np_path + "keras44_01_y_test.npy")
    print("파일 로드 완료!")
except FileNotFoundError:
    print("지정된 경로에서 NPY 파일을 찾을 수 없습니다. 더미 데이터를 생성합니다.")
    x_train = np.random.rand(17500, 50, 50, 3).astype('float32') * 255
    y_train = np.random.randint(0, 2, 17500)
    x_test = np.random.rand(7500, 50, 50, 3).astype('float32') * 255
    y_test = np.random.randint(0, 2, 7500)

end = time.time()
print("데이터 로드 시간:", round(end - start, 2), "초")

print("--- 초기 데이터 형태 ---")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("-" * 30)

# 이미지의 픽셀 값을 0-1 사이로 정규화합니다. 이 과정은 모델 학습에 필수적입니다.
x_train = x_train / 255.0
x_test = x_test / 255.0

print("--- 전처리 완료된 데이터 형태 ---")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("-" * 30)

# --- 3. 모델 정의 및 반복 학습 ---
# 사용하려는 사전 학습 모델 목록.
# 모델은 (50, 50, 3) 입력 형태에 맞게 설정됩니다.
model_list = [
    VGG16, ResNet50, ResNet101, DenseNet121, MobileNetV2, EfficientNetB0
]

# 결과를 저장할 리스트를 생성합니다.
results = []

for model_builder in model_list:
    # 모델 이름을 가져옵니다.
    model_name = model_builder.__name__
    print(f"\n===== 모델 평가 시작: {model_name} =====")

    # 기본 모델을 불러옵니다.
    try:
        base_model = model_builder(
            weights='imagenet',
            include_top=False,
            input_shape=(50, 50, 3) # 입력 형태를 (50, 50, 3)으로 수정
        )
    except Exception as e:
        print(f"모델 {model_name}은 50x50 입력에 적합하지 않습니다. 다음 모델로 넘어갑니다. 에러: {e}")
        continue

    # 기본 모델의 가중치를 동결(freeze)합니다.
    base_model.trainable = False

    # 새로운 Sequential 모델을 구축합니다.
    model = Sequential([
        base_model,
        # GlobalAveragePooling2D 레이어를 사용해 특징 맵을 단일 벡터로 변환합니다.
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        # 이진 분류에는 1개의 뉴런과 'sigmoid' 활성화 함수를 사용합니다.
        Dense(1, activation='sigmoid')
    ])

    # 모델을 컴파일합니다.
    # 이진 분류에는 'binary_crossentropy' 손실 함수를 사용합니다.
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # EarlyStopping 콜백을 설정하여 과적합을 방지하고 최적의 가중치를 저장합니다.
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )

    # 모델 학습
    print(f"--- 학습 시작: {model_name} ---")
    start_time = time.time()
    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )
    end_time = time.time()

    # 모델 평가
    print(f"\n--- 평가 시작: {model_name} ---")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    # 예측 결과와 실제 레이블 비교
    y_pred_probs = model.predict(x_test)
    y_pred_labels = (y_pred_probs > 0.5).astype(int)
    accuracy_manual = accuracy_score(y_test, y_pred_labels)

    # 결과를 딕셔너리로 저장합니다.
    results.append({
        '모델': model_name,
        '테스트 손실': f'{loss:.4f}',
        '테스트 정확도': f'{acc:.4f}',
        '학습 시간': f'{end_time - start_time:.2f}초'
    })

    # 결과 출력
    print(f"모델: {model_name}")
    print(f"  테스트 손실: {loss:.4f}")
    print(f"  테스트 정확도 (evaluate): {acc:.4f}")
    print(f"  테스트 정확도 (accuracy_score): {accuracy_manual:.4f}")
    print(f"  학습 시간: {end_time - start_time:.2f}초")
    print("=" * (28 + len(model_name)))

# 모든 모델의 결과가 끝나면 표로 정리하여 출력합니다.
print("\n===== 전체 모델 성능 요약 =====")
df_results = pd.DataFrame(results)
print(df_results)
print("=" * 30)


# ===== 전체 모델 성능 요약 =====
#                모델  테스트 손실 테스트 정확도    학습 시간
# 0              VGG16    0.6082      0.9900        52.79초
# 1           ResNet50    0.7630      0.0000        38.85초   # 모델 동결(Freezing) 문제: 
# 2          ResNet101    0.0430      1.0000        61.57초   # base_model.trainable = False로 인해 
# 3        DenseNet121    0.9519      0.2000        65.87초   # 모델의 가중치가 업데이트되지 않아
# 4        MobileNetV2    0.4563      0.9000        41.73초   # 학습이 전혀 이루어지지 않았을 수 있음.
# 5     EfficientNetB0    0.7022      0.0000       107.79초
# ==============================