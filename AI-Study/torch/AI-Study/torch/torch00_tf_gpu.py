import tensorflow as tf
print("tensorflow version :", tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print("GPU TRUE")
else:
    print("GPU FALSE")

# CUDA VERSION
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print("CUDA version :", cuda_version)

# CUDNN VERSION
cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print("cuDNN version :", cudnn_version)

# tensorflow version : 2.10.0
# GPU TRUE
# CUDA version : 64_112
# cuDNN version : 64_8 
# (wj313) PS C:\Study25> 
