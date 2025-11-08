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

# ---------------------------
# 1. 데이터 로드 및 분리
# ---------------------------
np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
x1_train = np.load(np_path + "keras46_07_x_train.npy")
y1_train = np.load(np_path + "keras46_07_y_train.npy")
end = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    x1_train, y1_train, test_size=0.3, random_state=333, shuffle=True)

# ---------------------------
# 2. 데이터 형태 변형 및 정규화
# ---------------------------
x_train = x_train.reshape(-1, 250, 250, 3)
x_test = x_test.reshape(-1, 250, 250, 3)



print("--- Initial Data Shapes ---")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("-" * 30)

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0


print("--- Preprocessed Data Shapes ---")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("-" * 30)

# --- 3. Model Definitions ---
# List of pre-trained models to evaluate.
# All models will use weights from 'imagenet' and the same input shape.
model_list = [
    VGG16, ResNet50, ResNet101, DenseNet121, MobileNetV2, EfficientNetB0
]

# --- 4. Loop Through Models for Training and Evaluation ---
for model_builder in model_list:
    # Instantiate the base model
    base_model = model_builder(
        weights='imagenet',
        include_top=False,
        input_shape=(250, 250, 3)
    )
    # Freeze the convolutional base
    base_model.trainable = False

    model_name = base_model.name
    print(f"\n===== Evaluating Model: {model_name} =====")

    # Create the new model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid') 
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Define EarlyStopping
    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5, # Shorter patience for faster iteration
        verbose=1,
        restore_best_weights=True
    )

    # Train the model
    print(f"--- Training {model_name} ---")
    start_time = time.time()
    model.fit(
        x_train, y_train,
        epochs=20, # Early stopping will likely stop it sooner
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )
    end_time = time.time()

    
    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten() # flatten converts [[1],[0]] to [1,0]

    # Evaluate the model
    print(f"\n--- Evaluating {model_name} ---")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    # Print results for the current model
    print(f"Model: {model_name}")
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Training Time: {end_time - start_time:.2f} seconds")
    print("=" * (28 + len(model_name)))
    
# --- Evaluating vgg16 ---
# Model: vgg16
#   Test Loss: 0.6883
#   Test Accuracy: 0.5690
#   Training Time: 88.08 seconds
# =================================

# --- Evaluating resnet50 ---
# Model: resnet50
#   Test Loss: 1.9496
#   Test Accuracy: 0.4310
#   Training Time: 55.71 seconds
# ====================================

# --- Evaluating resnet101 ---
# Model: resnet101
#   Test Loss: 3.1701
#   Test Accuracy: 0.4310
#   Training Time: 84.10 seconds
# =====================================

# --- Evaluating densenet121 ---
# Model: densenet121
#   Test Loss: 0.7835
#   Test Accuracy: 0.4441
#   Training Time: 119.61 seconds
# =======================================

# --- Evaluating mobilenetv2_1.00_224 ---
# Model: mobilenetv2_1.00_224
#   Test Loss: 0.7211
#   Test Accuracy: 0.4361
#   Training Time: 44.98 seconds
# ================================================

# --- Evaluating efficientnetb0 ---
# Model: efficientnetb0
#   Test Loss: 0.6838
#   Test Accuracy: 0.5690
#   Training Time: 57.82 seconds
# ==========================================