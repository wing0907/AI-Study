from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet169, DenseNet121
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications import Xception

model_list = [VGG16(), VGG19(),
    ResNet50(), ResNet50V2(), ResNet101(), ResNet101V2(), ResNet152(), ResNet152V2(),
    DenseNet121(), DenseNet169(), DenseNet201(),
    InceptionV3(), InceptionResNetV2(),
    MobileNet(), MobileNetV2(), MobileNetV3Small(), MobileNetV3Large(),
    NASNetMobile(), NASNetLarge(),
    EfficientNetB0(), EfficientNetB1(), EfficientNetB2(),
    Xception()
    ]

# for문 써서 결과 출력하기
for model in model_list:
    model.trainable = False
    
    print("="*100)
    print("모델명:", model.name)
    print("전체 가중치 갯수:", len(model.weights))
    print("훈련 가능 갯수:", len(model.trainable_weights))
    