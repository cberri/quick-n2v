import tensorflow as tf
import os

	



gpu = int(0)

if gpu==(0):
		# Count how many GPUs are available
		gpus = tf.config.experimental.list_physical_devices('GPU')
		gpu = 0
		for n_gpu in gpus:
			gpu += 1

		# Set the last GPU in the GPUs list
		#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu-1)  # "0, 1, 2, 3"
else:
		# Set the last GPU in the GPUs list
		#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # "0, 1, 2, 3"
	
	# Solve tf memory issue/ take only the memory necesary
	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	#session = InteractiveSession(config=config)


	print("n2v runs on GPU " + str(gpu))
