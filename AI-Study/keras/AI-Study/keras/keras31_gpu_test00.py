import tensorflow as tf
print(tf.__version__) # 2.9.3           tensorflow는 2.10 버전까지는 gpu와 cpu 버전이 따로 나누어져있다

gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

