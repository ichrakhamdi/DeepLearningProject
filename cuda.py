import tensorflow as tf

print("CUDA Available:", tf.test.is_built_with_cuda())

print("cuDNN Version:", tf.sysconfig.get_libcudnn_version())

 