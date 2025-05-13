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
from tensorflow.keras.datasets import cifar100
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. Setup and Configuration ---
SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. Data Loading and Preprocessing ---
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("--- Initial Data Shapes ---")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
print("-" * 30)

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-Hot Encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 100)
y_test = tf.keras.utils.to_categorical(y_test, 100)

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
        input_shape=(32, 32, 3)
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
        Dense(100, activation='softmax') # 100 classes for CIFAR-100
    ])

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
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
        epochs=50, # Early stopping will likely stop it sooner
        batch_size=128,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )
    end_time = time.time()

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
#   Test Loss: 2.6300
#   Test Accuracy: 0.3454
#   Training Time: 116.08 seconds
# =================================

# --- Evaluating resnet50 ---
# Model: resnet50
#   Test Loss: 6.2410
#   Test Accuracy: 0.0112
#   Training Time: 52.14 seconds
# ====================================

# --- Evaluating resnet101 ---
# Model: resnet101
#   Test Loss: 5.1244
#   Test Accuracy: 0.0134
#   Training Time: 84.99 seconds
# =====================================

# --- Evaluating densenet121 ---
# Model: densenet121
#   Test Loss: 2.4392
#   Test Accuracy: 0.3814
#   Training Time: 168.61 seconds
# =======================================

# --- Evaluating mobilenetv2_1.00_224 ---
# Model: mobilenetv2_1.00_224
#   Test Loss: 3.8018
#   Test Accuracy: 0.1332
#   Training Time: 66.49 seconds
# ================================================

# --- Evaluating efficientnetb0 ---
# Model: efficientnetb0
#   Test Loss: 4.6053
#   Test Accuracy: 0.0100
#   Training Time: 203.99 seconds
# ==========================================