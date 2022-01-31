import argparse
import datetime
import os
import tifffile as tiff
from tqdm import tqdm
from vtools import generate_args, prepare_training_data, train_model
from matplotlib.image import imread, imsave
from n2v.models import N2V
import numpy as np
import shutil
import tempfile
import re
import subprocess as sp

# Fix the out of memory problem
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession

'''
Last update 2020-09-16

Example:
python onimagestunningModify.py --target '/media/math-clinic/FastData/Data/n2V/Varun Ramani/2020-06-30/training/speed6' 
--baseDir '/media/math-clinic/FastData/Data/n2V/Varun Ramani/2020-06-30/training/model6' --dims XY --stack n --train y

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # The GPU id to use, usually either "0", "1". "2", "3";

'''

# Input user setting
parser = argparse.ArgumentParser(description='quick-n2v')
parser.add_argument('--target', metavar='target', type=str, default='video_images',
                    help='path to the target directory containing the input images')
parser.add_argument('--output', metavar='output', type=str, default=None,
                    help='output directory full path')
parser.add_argument('--baseDir', metavar='baseDir', type=str, default='models',
                    help='directory path to store the trained models and the configurations.')
parser.add_argument('--train', metavar='train', type=str, default='n',
                    help='force train? y or n (default=n)')

parser.add_argument('--dims', metavar='dims', type=str, default='XY',
                    help='dimensions of your data, can include: X,Y,Z,C (default=YX)')
parser.add_argument('--fileName', metavar='fileName', type=str, default='*.tif',
                    help='file name ending (default=*.tif)')
parser.add_argument('--clipping', metavar='clipping', type=str, default='minmax',
                    help='clipping approach (imageclip,minmax,zeromax,0255 default=minmax) \n \t imageclip: make output image in the same range input. \n \t minmax: apply min max normalization and makes between 0 and 1. \n \t zeromax: clip between 0 and max of input image, 0255: clip the prediction between 0 and 255.')
parser.add_argument('--formatOut', metavar='formatOut', type=str, default='.tif',
                    help='format of the output. Noticed that when png and XY it makes a RGB image in gray scale (png, .tif default: .tif)')
parser.add_argument('--saveInputs', metavar='', type=str, default='n',
                    help='save inputs to the network that maybe have been converted (y, n default: n)')
parser.add_argument('--stack', metavar='', type=str, default='n',
                    help='process images as stack of 2D images (y, n default: n)')

parser.add_argument("--name", metavar='name', help="name of your network default=N2V", default='N2V')
parser.add_argument("--validationFraction",
                    help="Fraction of data you want to use for validation (percent default=10.0)",
                    default=10.0, type=float)  # 5
parser.add_argument("--patchSizeXY", metavar='', help="XY-size of your training patches (default=64)", default=64,
                    type=int)
parser.add_argument("--patchSizeZ", help="Z-size of your training patches (default=64)", default=64, type=int)
parser.add_argument("--epochs", help="number of training epochs (default=100)", default=100, type=int)
parser.add_argument("--stepsPerEpoch", help="number training steps per epoch (default=400)", default=400, type=int)
parser.add_argument("--batchSize", help="size of your training batches (default=64)", default=64, type=int)
parser.add_argument("--netDepth", help="depth of your U-Net (default=2)", default=2, type=int)
parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer (default=3)", default=3, type=int)
parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V (default=1.6)", default=1.6,
                    type=float)
parser.add_argument("--learningRate", help="initial learning rate (default=0.0004)", default=0.0004, type=float)
parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer (default=32)",
                    default=32,
                    type=int)
parser.add_argument("--gpu", help="default gpu is 0", default='0', type=int)
args = parser.parse_args()


###########################################################################
# Set the GPU for running n2v
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]

    COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

    return memory_used_values, memory_total_values


def use_selected_card(gpu, memory_used_values, memory_total_values):
    # Use the first free GPU
    if gpu == 0:
        array_gpu = []  # [0 for n_gpu in gpus]

        # Print the list of available GPUs
        for i in range(len(memory_total_values)):

            free_memory = memory_total_values[i] - memory_used_values[i]
            print('>> Max GPU' + str(i), ' memory:', memory_total_values[i], ' Used memory: ', memory_used_values[i],
                  ' Free memory: ', free_memory)

            if (free_memory > (memory_total_values[i] * 0.2)):
                array_gpu.append(i)

        # Set first available GPU in the list
        default_gpu = array_gpu[-len(array_gpu)]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(default_gpu)  # "0, 1, 2, 3"

    else:
        # Choose one GPU / Never the first
        default_gpu = gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(default_gpu)  # "0, 1, 2, 3"

    # Solve tf out of memory problem
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    print(">> Running on GPU " + str(default_gpu))


# Set GPU
memory_used_values, memory_total_values = get_gpu_memory()
use_selected_card(args.gpu, memory_used_values, memory_total_values)


# Modified by Carlo Beretta (add input file name in the output directory file name)
def create_output_directory(output_path):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.target), 'denoised_images_' + str(
            os.path.basename(os.path.normpath(args.target))) + '_' + str(datetime.datetime.now()).replace(' ',
                                                                                                          '_').replace(
            '.', 'p').replace(':', 'T'))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path


def clip(pred, lb, ub):
    if args.clipping == 'zeromax' or args.clipping == '0255':
        pred = pred.copy()
        pred[pred < 0] = 0
    if args.clipping == '0255':
        pred = pred.copy()
        pred[pred > 255] = 255
        # avoids normalization
        return pred
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = (ub - lb) * pred + lb
    return pred


def create_unravel_folder(images_path):
    '''
	Creates an unraveled folder containing all the images from a image directory of stacked images_path
	args:
	images_path:Path to stacked images_path
	return:
	temporary path of stacked images
	'''
    temp_dir = tempfile.mkdtemp(prefix='quick-n2v')
    print('Temporary files are stored in: ', temp_dir)
    for f in tqdm(os.listdir(images_path)):
        if os.path.isfile(os.path.join(images_path, f)) and f.endswith('.tif'):
            img = tiff.imread(os.path.join(images_path, f))
            for ii, im in enumerate(img):
                f_out = os.path.join(temp_dir, f.split('.tif')[0] + '.' + str(ii) + '.tif')
                tiff.imsave(f_out, im)
    return temp_dir


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
		"z23a" -> ["z", 23, "a"]
	"""
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
	"""
    l.sort(key=alphanum_key)


def concatenate_unravel_folder(images_path, output_path):
    '''
	concatenates an unraveled stacked image again into stacked images_path
	args:
	images_path: path to the unraveled images_path
	output_path: path to directory to save stacked images
	'''
    # Collect all files relevant to the stacking
    all_files_relevant = []
    print('Find relevant files in ', images_path)
    for f in tqdm(os.listdir(images_path)):

        if os.path.isfile(os.path.join(images_path, f)) and f.endswith('.tif'):
            all_files_relevant.append(f)
    sort_nicely(all_files_relevant)

    # create stacks once the images are sorted
    n0 = -1
    image_paths = []
    print('Concatenating images...')
    N = len(all_files_relevant) - 1
    for i, f in tqdm(enumerate(all_files_relevant)):
        n1 = int(f.split('.')[-2])
        image_paths.append(os.path.join(images_path, f))
        if n1 < n0 or i == N:
            # generate stacked image
            file_name_stack = os.path.basename(image_paths[0]).split('.0.tif')[0] + '.tif'
            f_out = os.path.join(output_path, file_name_stack)
            imgs = np.array([tiff.imread(img_path) for img_path in image_paths])
            tiff.imsave(f_out, imgs)
            n0 = -1
            image_paths = []
        else:
            # sequence continues
            n0 = n1


def denoise_images(images_path, output_path):
    '''
	Denoise images in a path and store the resutl into output path
	:param images_path:
	:param output_path:
	:return:
	'''
    model_name = 'N2V'
    basedir = args.baseDir
    # basedir = 'models'

    model = N2V(config=None, name=model_name, basedir=basedir)
    warning_print = False
    print('DENOISING......')
    for f in tqdm(os.listdir(images_path)):
        if os.path.isfile(os.path.join(images_path, f)) and f.endswith(args.fileName.replace('*', '')):
            img = imread(os.path.join(images_path, f))

            # if args.dims = N means Not Defined. Then the program decides which is the one.
            if len(img.shape) == 2 and args.dims == 'N':
                axes = 'YX'
            elif len(img.shape) == 3 and args.dims == 'N':
                axes = 'YXC'
            elif len(img.shape) == 4 and args.dims == 'N':
                axes = 'XYZ'
            elif len(img.shape) == 5 and args.dims == 'N':
                axes = 'ZYXC'
            else:
                axes = args.dims
            # Ensures that the input has 3 channels if read image axes are XYC or YXC
            if 'C' in axes and img.shape[-1] == 4:
                if not warning_print:
                    print('Warning: alpha channels will be removed from all images.')
                    warning_print = True
                    print('Org. shape: ', img.shape, 'New shape:', img[..., :3].shape)
                img = img[..., :3]
            if len(axes) == 2 and not len(img.shape) == 2:
                if not warning_print:
                    print('Warning: Enfoced 2 channels. 3 channels detected. All images will be converted.')
                    warning_print = True
                    print('Org. shape: ', img.shape, 'New shape:', img[..., 0].shape)
                img = img[..., 0]  # taking the red channel
            pred = model.predict(img, axes=axes)
            if args.saveInputs == 'y':
                f_out_d = os.path.join(output_path,
                                       'Denoised-' + f.replace(args.fileName.replace('*', ''), args.formatOut))
                f_out_s = os.path.join(output_path,
                                       'Input-' + f.replace(args.fileName.replace('*', ''), args.formatOut))
            else:
                f_out_d = os.path.join(output_path, f.replace(args.fileName.replace('*', ''), args.formatOut))
            print('pred.max(): ', pred.max(), 'pred.min()', pred.min())
            print('img.max(): ', img.max(), 'img.min()', img.min())
            if args.clipping == 'imageclip':
                ub = img.max()
                lb = img.min()
            elif args.clipping == 'zeromax':
                ub = img.max()
                lb = 0
            elif args.clipping == 'minmax':
                ub = 1
                lb = 0
            elif args.clipping == '0255':
                ub = 1
                lb = 0
            else:
                raise Exception(
                    'Invalid input value clipping not supported.' + args.clipping + '. Check --help for datails.')
            if args.formatOut == '.png':
                print('saving file denoised : ', f_out_d)
                imsave(f_out_d, clip(pred, lb, ub), cmap='gray')
                if args.saveInputs == 'y':
                    print('saving file input to network: ', f_out_s)
                    imsave(f_out_s, clip(img, lb, ub), cmap='gray')
            elif args.formatOut == '.tif':
                print('saving file denoised : ', f_out_d)
                tiff.imsave(f_out_d, clip(pred, lb, ub))
                if args.saveInputs == 'y':
                    print('saving file input to network: ', f_out_s)
                    tiff.imsave(f_out_s, clip(img, lb, ub))
            else:
                raise Exception('Supported output formats png and tif. Other format not supported' + args.formatOut)


# Creating the path of denoised images
output_path = create_output_directory(args.output)
print('Output path is: ', output_path)

if args.stack == 'y':
    print('WARNING: Stack input selected. temporary files will be generated')
    unravel_path = create_unravel_folder(args.target)
    unravel_output_path = create_output_directory(os.path.join(unravel_path, 'unraveled_denoised'))
    if args.train == 'y' or not os.path.exists(
            os.path.join(args.baseDir, 'N2V/') + 'weights_best.h5'):  # ('models/N2V/weights_best.h5'):
        training_args = generate_args(data_path=args.target, fileName=args.fileName, dims=args.dims,
                                      baseDir=args.baseDir, name=args.name, validationFraction=args.validationFraction,
                                      patchSizeXY=args.patchSizeXY, patchSizeZ=args.patchSizeZ, epochs=args.epochs,
                                      stepsPerEpoch=args.stepsPerEpoch, batchSize=args.batchSize,
                                      netDepth=args.netDepth,
                                      netKernelSize=args.netKernelSize, n2vPercPix=args.n2vPercPix,
                                      learningRate=args.learningRate, unet_n_first=args.unet_n_first)
        model, X, X_val = prepare_training_data(training_args)
        history = train_model(model, X, X_val)
    # apply on video
    denoise_images(unravel_path, unravel_output_path)
    # concatenate images in output path
    concatenate_unravel_folder(unravel_output_path, output_path)
    print("Removing the temp folder: ", unravel_path)
    shutil.rmtree(unravel_path)

# Export n2v model for ImageJ/Fiji
# As it is the model.yaml file is not created and Fiji through an error message
# model.export_TF()

else:
    if args.train == 'y' or not os.path.exists(
            os.path.join(args.baseDir, 'N2V/') + 'weights_best.h5'):  # ('models/N2V/weights_best.h5'):
        training_args = generate_args(data_path=args.target, fileName=args.fileName, dims=args.dims,
                                      baseDir=args.baseDir, name=args.name, validationFraction=args.validationFraction,
                                      patchSizeXY=args.patchSizeXY, patchSizeZ=args.patchSizeZ, epochs=args.epochs,
                                      stepsPerEpoch=args.stepsPerEpoch, batchSize=args.batchSize,
                                      netDepth=args.netDepth,
                                      netKernelSize=args.netKernelSize, n2vPercPix=args.n2vPercPix,
                                      learningRate=args.learningRate, unet_n_first=args.unet_n_first)
        model, X, X_val = prepare_training_data(training_args)
        history = train_model(model, X, X_val)

        # Export n2v model for ImageJ/Fiji
        # As it is the model.yaml file is not created and Fiji through an error message
        model.export_TF()

    # apply on video
    denoise_images(args.target, output_path)

print('>> Applied model path: ', os.path.join(args.baseDir, 'N2V/') + 'weights_best.h5')
