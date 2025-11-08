"""
keras46, 47을 참고하여, 남자 여자 사진에 노이즈를 주고,
내 사진에도 노이즈를 추가,
오토 인코더(CAE)로 피부 미백 훈련 가중치를 만든다.
그 가중치로 내 사진을 예측해서 피부 미백    
"""
# -*- coding: utf-8 -*-
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────
# 0) 환경/경로 설정
# ─────────────────────────────────────────────────────────
SEED = 190
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ✅ 256x256로 변경 (스킵 연결 크기 불일치 해결)
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 200

# ✅ 남/여 사진이 합쳐진 폴더 (하위 male/female 없이 이미지만 있음)
TRAIN_DIR = r'C:\Study25\_data\_save_image\08_men_women'
# ✅ 내 사진 폴더
MY_DIR    = r'C:\Study25\_data\image\me'

os.makedirs('./_save/cae_whitening/', exist_ok=True)
WEIGHT_PATH = './_save/cae_whitening/cae_whitening_best.h5'

# ─────────────────────────────────────────────────────────
# 1) 유틸: 파일 로드/전처리/증강
# ─────────────────────────────────────────────────────────
def list_images(root_dir, exts=('.jpg','.jpeg','.png','.bmp','.webp')):
    files = []
    for d, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(exts):
                files.append(os.path.join(d, fn))
    return sorted(files)

def load_image(path, target_size=IMG_SIZE):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0   # [0,1]
    return arr.astype('float32')

def add_gaussian_noise(x, std=0.06):
    noise = np.random.normal(0, std, size=x.shape).astype('float32')
    return np.clip(x + noise, 0.0, 1.0)

def to_hsv_and_whiten(x, v_gain=1.15, s_scale=0.90, gamma=0.90):
    """
    x: float32 [0,1], RGB
    - HSV 변환 후 V(밝기) ↑, S(채도) ↓
    - 감마 보정(γ<1)으로 추가 밝기
    """
    x_tf = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    hsv  = tf.image.rgb_to_hsv(x_tf)
    h, s, v = tf.split(hsv, 3, axis=-1)
    v = tf.clip_by_value(v * v_gain, 0.0, 1.0)
    s = tf.clip_by_value(s * s_scale, 0.0, 1.0)
    hsv_adj = tf.concat([h, s, v], axis=-1)
    rgb_adj = tf.image.hsv_to_rgb(hsv_adj)
    rgb_gamma = tf.clip_by_value(tf.pow(rgb_adj, gamma), 0.0, 1.0)
    return rgb_gamma[0].numpy().astype('float32')

# ─────────────────────────────────────────────────────────
# 2) tf.data 파이프라인 (입력=노이즈, 타깃=미백)
# ─────────────────────────────────────────────────────────
def build_pairs(image_paths, val_split=0.1):
    n = len(image_paths)
    assert n > 0, f"훈련 이미지가 없습니다: {TRAIN_DIR}"
    idxs = np.arange(n); np.random.shuffle(idxs)
    vs = int(n * val_split)
    val_idx, tr_idx = idxs[:vs], idxs[vs:]

    def make_ds(idxs_subset, shuffle=True):
        def gen():
            for i in idxs_subset:
                x = load_image(image_paths[i])                # clean
                x_in  = add_gaussian_noise(x, std=0.06)       # noisy input
                x_tgt = to_hsv_and_whiten(x, 1.15, 0.90, 0.90)# whitened target
                yield x_in, x_tgt
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            ),
        )
        if shuffle:
            ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return make_ds(tr_idx, True), make_ds(val_idx, False)

# ─────────────────────────────────────────────────────────
# 3) 모델: 컨볼루션 오토인코더 (소형 U-Net 스타일)
# ─────────────────────────────────────────────────────────
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))
psnr_metric.__name__ = 'psnr'

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
ssim_metric.__name__ = 'ssim'

def build_cae(input_shape=(256,256,3)):
    inp = layers.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x1 = layers.BatchNormalization()(x1)
    p1 = layers.MaxPooling2D()(x1)  # 256 -> 128

    x2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    x2 = layers.BatchNormalization()(x2)
    p2 = layers.MaxPooling2D()(x2)  # 128 -> 64

    x3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    x3 = layers.BatchNormalization()(x3)
    p3 = layers.MaxPooling2D()(x3)  # 64 -> 32

    bott = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)

    # Decoder
    u3 = layers.UpSampling2D()(bott)           # 32 -> 64
    u3 = layers.Concatenate()([u3, x3])
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    u3 = layers.BatchNormalization()(u3)

    u2 = layers.UpSampling2D()(u3)             # 64 -> 128
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.BatchNormalization()(u2)

    u1 = layers.UpSampling2D()(u2)             # 128 -> 256
    u1 = layers.Concatenate()([u1, x1])
    u1 = layers.Conv2D(32, 3, padding='same', activation='relu')(u1)
    u1 = layers.BatchNormalization()(u1)

    out = layers.Conv2D(3, 1, activation='sigmoid')(u1)

    model = models.Model(inp, out, name='CAE_Whitening_256')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mae',  # L1이 복원계열에서 부자연스러운 blur 감소에 유리
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.MeanSquaredError(name='mse'),
            psnr_metric,
            ssim_metric,
        ],
    )
    return model

# ─────────────────────────────────────────────────────────
# 4) 학습
# ─────────────────────────────────────────────────────────
all_train_imgs = list_images(TRAIN_DIR)
print(f"train images: {len(all_train_imgs)}")
assert len(all_train_imgs) > 0, f"이미지가 없습니다. 경로/확장자를 확인하세요: {TRAIN_DIR}"

train_ds, val_ds = build_pairs(all_train_imgs, val_split=0.1)

model = build_cae(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.summary()

es  = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', save_best_only=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[es, mcp, rlr],
    verbose=1
)

print("\nBest weights saved to:", WEIGHT_PATH)

# ─────────────────────────────────────────────────────────
# 5) 내 사진: 노이즈 → 미백 예측 → 저장/시각화
# ─────────────────────────────────────────────────────────
def first_image_in(folder):
    lst = list_images(folder)
    assert len(lst) > 0, f"내 사진이 없습니다: {folder}"
    return lst[0]

me_path  = first_image_in(MY_DIR)
me_clean = load_image(me_path, target_size=IMG_SIZE)
me_noisy = add_gaussian_noise(me_clean, std=0.06)

pred = model.predict(me_noisy[None, ...], verbose=0)[0]  # [H,W,3], [0,1]

# 저장
ts = datetime.datetime.now().strftime('%m%d_%H%M%S')
plt.imsave(os.path.join(MY_DIR, f'me_clean_{ts}.png'),   me_clean)
plt.imsave(os.path.join(MY_DIR, f'me_noisy_{ts}.png'),   me_noisy)
plt.imsave(os.path.join(MY_DIR, f'me_whitened_{ts}.png'), pred)

# 시각화
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(me_clean);  plt.title('원본');      plt.axis('off')
plt.subplot(1,3,2); plt.imshow(me_noisy);  plt.title('노이즈 입력'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(pred);      plt.title('CAE 미백 출력'); plt.axis('off')
plt.tight_layout(); plt.show()
