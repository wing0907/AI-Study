# m36_1에서 뽑은 4가지 결과로 4개의 모델 만들기
# input_shape = 
# (70000, 154)
# (70000, 331)
# (70000, 486)
# (70000, 713)

# 힌트 :
num = [154, 331, 486, 713, 784]

from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. 데이터 로딩 및 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0).reshape(70000, -1)  # (70000, 784)
y = np.concatenate([y_train, y_test], axis=0)
y = to_categorical(y)  # 원-핫 인코딩 (10 클래스)

# 2. PCA 차원 수 리스트
num_components_list = [154, 331, 486, 713, 784]

# 3. DNN 모델 정의 함수
def build_dnn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. 반복 실행
for n in num_components_list:
    print(f"\n▶ n_components = {n}")
    
    # PCA
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x)
    
    # train/test split
    x_train_pca, x_test_pca, y_train_split, y_test_split = train_test_split(
        x_pca, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    # DNN 모델 생성 및 학습
    model = build_dnn(n)
    model.fit(x_train_pca, y_train_split, epochs=10, batch_size=128,
              validation_split=0.2, verbose=0)
    
    # 평가
    loss, acc = model.evaluate(x_test_pca, y_test_split, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")



# ▶ n_components = 331
# Test Accuracy: 0.9149

# ▶ n_components = 486
# Test Accuracy: 0.9098

# ▶ n_components = 713
# Test Accuracy: 0.9238

# ▶ n_components = 784
# Test Accuracy: 0.9249

# ▶ n_components = 154
# Test Accuracy: 0.8930

# 쓰레기 데이터는 압축시키는게 나을 수 있다.