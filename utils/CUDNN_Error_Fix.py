# Solve the error: E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
