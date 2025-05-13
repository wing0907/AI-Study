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

# --- 2. 데이터 로드 및 전처리 ---
# 사용자의 로컬 NPY 파일을 불러옵니다.
# NOTE: 파일 경로를 사용자의 환경에 맞게 수정해야 합니다.
np_path = 'C:\Study25\_data\_save_npy\\'

start = time.time()
try:
    print(f"'{np_path}keras46_05_x_train.npy' 파일을 불러오는 중...")
    x1_train = np.load(np_path + "keras46_05_x_train.npy")
    print(f"'{np_path}keras46_05_y_train.npy' 파일을 불러오는 중...")
    y1_train = np.load(np_path + "keras46_05_y_train.npy")
    print("파일 로드 완료!")
except FileNotFoundError:
    print("지정된 경로에서 NPY 파일을 찾을 수 없습니다. 더미 데이터를 생성합니다.")
    x1_train = np.random.rand(2048, 100, 100, 3).astype('float32') * 255
    y1_train = np.zeros((2048, 3), dtype=np.int32)
    y1_train[np.arange(2048), np.random.randint(0, 3, 2048)] = 1

end = time.time()
print("데이터 로드 시간:", round(end-start, 2), "초")

# 데이터 분할 (train/test)
x_train, x_test, y_train, y_test = train_test_split(
    x1_train, y1_train, test_size=0.3, random_state=SEED, shuffle=True
)

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
# 모델은 (100, 100, 3) 입력 형태에 맞게 설정됩니다.
model_list = [
    VGG16, ResNet50, ResNet101, DenseNet121, MobileNetV2, EfficientNetB0
]

for model_builder in model_list:
    # 모델 이름을 가져옵니다.
    model_name = model_builder.__name__
    print(f"\n===== 모델 평가 시작: {model_name} =====")

    # 기본 모델을 불러옵니다.
    try:
        base_model = model_builder(
            weights='imagenet',
            include_top=False,
            input_shape=(100, 100, 3)
        )
    except Exception as e:
        print(f"모델 {model_name}은 100x100 입력에 적합하지 않습니다. 다음 모델로 넘어갑니다. 에러: {e}")
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
        # 3개의 클래스를 가지므로 3개의 뉴런과 'softmax' 활성화 함수를 사용합니다.
        Dense(3, activation='softmax')
    ])

    # 모델을 컴파일합니다.
    # 다중 클래스 분류에는 'categorical_crossentropy' 손실 함수를 사용합니다.
    model.compile(
        loss='categorical_crossentropy',
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
    
    # model.evaluate는 [손실, 정확도] 리스트를 반환합니다.
    # 이 값을 각각 loss와 acc 변수에 할당하여 오류를 해결했습니다.
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    # 예측 결과와 실제 레이블 비교
    y_pred_probs = model.predict(x_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    accuracy_manual = accuracy_score(y_test_labels, y_pred_labels)

    # 결과 출력
    print(f"모델: {model_name}")
    print(f"  테스트 손실: {loss:.4f}")
    print(f"  테스트 정확도 (evaluate): {acc:.4f}")
    print(f"  테스트 정확도 (accuracy_score): {accuracy_manual:.4f}")
    print(f"  학습 시간: {end_time - start_time:.2f}초")
    print("=" * (28 + len(model_name)))
    
# --- 평가 시작: VGG16 ---
# 모델: VGG16
#   테스트 손실: 1.2923
#   테스트 정확도 (evaluate): 0.3919
#   테스트 정확도 (accuracy_score): 0.3919
#   학습 시간: 25.49초
# =================================


# --- 평가 시작: ResNet50 ---
# 모델: ResNet50
#   테스트 손실: 1.9006
#   테스트 정확도 (evaluate): 0.3919
#   테스트 정확도 (accuracy_score): 0.3919
#   학습 시간: 11.36초
# ====================================


# --- 평가 시작: ResNet101 ---
# 모델: ResNet101
#   테스트 손실: 1.5146
#   테스트 정확도 (evaluate): 0.3919
#   테스트 정확도 (accuracy_score): 0.3919
#   학습 시간: 14.53초
# =====================================


# --- 평가 시작: DenseNet121 ---
# 모델: DenseNet121
#   테스트 손실: 1.1498
#   테스트 정확도 (evaluate): 0.4325
#   테스트 정확도 (accuracy_score): 0.4325
#   학습 시간: 17.91초
# =======================================


# --- 평가 시작: MobileNetV2 ---
# 모델: MobileNetV2
#   테스트 손실: 1.2881
#   테스트 정확도 (evaluate): 0.4325
#   테스트 정확도 (accuracy_score): 0.4325
#   학습 시간: 6.60초
# =======================================


# --- 평가 시작: EfficientNetB0 ---
# 모델: EfficientNetB0
#   테스트 손실: 1.0623
#   테스트 정확도 (evaluate): 0.4325
#   테스트 정확도 (accuracy_score): 0.4325
#   학습 시간: 12.85초
# ==========================================