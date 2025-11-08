import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


exit()





import tensorflow as tf
print(tf.__version__)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print('GPU사용')
else:
    print('CPU사용')



import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPU Devices:", tf.config.list_physical_devices('GPU'))