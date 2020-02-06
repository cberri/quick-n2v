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
from onimages import create_output_directory, generate_args, prepare_training_data, train_model, denoise_images, concatenate_unravel_folder, create_unravel_folder
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Denoise video with N2V')

    parser.add_argument('--target', metavar='target', type=str, default='video_images',
                    help='target directory containing png images full path')
    parser.add_argument('--output', metavar='output', type=str, default = None,
                    help='output directory full path')
    parser.add_argument('--baseDir', metavar='baseDir', type=str, default = 'models' ,
                    help='directory path to store the trained models and the configurations.')

    parser.add_argument('--train', metavar='train', type=str, default='n',
                    help='force train? y or n (default=n)')
    parser.add_argument('--dims', metavar='dims', type=str, default='XY',
                    help='dimensions of your data, can include: X,Y,Z,C (channel), T (time) (default=XY)')
    parser.add_argument('--fileName', metavar='fileName', type=str, default='*.tif',
                    help='file name ending (default=*.tif)')
    parser.add_argument('--clipping', metavar='clipping', type=str, default='minmax',
                    help='clipping approach (imageclip,minmax,zeromax,0255 default=minmax) \n \t imageclip: make output image in the same range input. \n \t minmax: apply min max normalization and makes between 0 and 1. \n \t zeromax: clip between 0 and max of input image, 0255: clip the prediction between 0 and 255.')
    parser.add_argument('--formatOut', metavar='formatOut', type=str, default='.tif',
                    help='format of the output. Noticed that when png and XY it makes a RGB image in gray scale (png, .tif default: .tif)')
    parser.add_argument('--saveInputs', metavar='saveInputs', type=str, default='n',
                    help='save inputs to the network that maybe have been converted (y, n default: n)')
    parser.add_argument('--stack', metavar='stack', type=str, default='n',
                    help='process images as stack of 2D images (y, n default: n)')
    parser.add_argument("--name", metavar='name', help="name of your network default=N2V", default='N2V')
    # parser.add_argument("--dataPath", help="The path to your training data")
    parser.add_argument("--validationFraction", help="Fraction of data you want to use for validation (percent default=5.0)",
                        default=5.0, type=float)
    parser.add_argument("--patchSizeXY", help="XY-size of your training patches default=64", default=64, type=int)
    parser.add_argument("--patchSizeZ", help="Z-size of your training patches default=64", default=64, type=int)
    parser.add_argument("--epochs", help="number of training epochs default=100", default=100, type=int)
    parser.add_argument("--stepsPerEpoch", help="number training steps per epoch default=400", default=400, type=int)
    parser.add_argument("--batchSize", help="size of your training batches default=64", default=64, type=int)
    parser.add_argument("--netDepth", help="depth of your U-Net default=2", default=2, type=int)
    parser.add_argument("--netKernelSize", help="Size of conv. kernels in first layer default=3", default=3, type=int)
    parser.add_argument("--n2vPercPix", help="percentage of pixels to manipulated by N2V default=1.6", default=1.6, type=float)
    parser.add_argument("--learningRate", help="initial learning rate default=0.0004", default=0.0004, type=float)
    parser.add_argument("--unet_n_first", help="number of feature channels in the first u-net layer default=32", default=32,
                        type=int)
    args = parser.parse_args()
    print(args)

    # Creating the path of denoised images
    output_path = create_output_directory(args.output)
    print('Output path is: ', output_path )
    if args.stack == 'y':
        print('WARNING: Stack input selected. temporary files will be generated')
        unravel_path = create_unravel_folder(args.target)
        unravel_output_path = create_output_directory(os.path.join(unravel_path,'unraveled_denoised'))
        if args.train=='y' or not os.path.exists('models/N2V/weights_best.h5'):
            training_args = generate_args(data_path=args.target, fileName=args.fileName, dims=args.dims,
                                          baseDir=args.baseDir, name=args.name, validationFraction=args.validationFraction,
                                          patchSizeXY=args.patchSizeXY, patchSizeZ=args.patchSizeZ, epochs=args.epochs,
                                          stepsPerEpoch=args.stepsPerEpoch, batchSize=args.batchSize, netDepth=args.netDepth,
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
    else:
        if args.train=='y' or not os.path.exists('models/N2V/weights_best.h5'):
            training_args = generate_args(data_path=args.target, fileName=args.fileName, dims=args.dims,
                                          baseDir=args.baseDir, name=args.name, validationFraction=args.validationFraction,
                                          patchSizeXY=args.patchSizeXY, patchSizeZ=args.patchSizeZ, epochs=args.epochs,
                                          stepsPerEpoch=args.stepsPerEpoch, batchSize=args.batchSize, netDepth=args.netDepth,
                                          netKernelSize=args.netKernelSize, n2vPercPix=args.n2vPercPix,
                                          learningRate=args.learningRate, unet_n_first=args.unet_n_first)
            model, X, X_val = prepare_training_data(training_args)
            history = train_model(model, X, X_val)
        # apply on video
        denoise_images(args.target, output_path)
