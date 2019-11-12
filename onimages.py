import argparse
import os
import tifffile as tiff
from tqdm import tqdm
from vtools import generate_args, prepare_training_data, train_model
from matplotlib.image import imread, imsave
from n2v.models import N2V
import numpy as np


parser = argparse.ArgumentParser(description='Denoise video with N2V')
parser.add_argument('--target', metavar='target', type=str, default='video_images',
                help='target directory containing png images full path')
parser.add_argument('--output', metavar='output', type=str, default = None,
                help='output directory full path')
parser.add_argument('--train', metavar='train', type=str, default='n',
                help='force train? y or n (default=n)')
parser.add_argument('--fileName', metavar='fileName', type=str, default='*.png',
                help='file name ending (default=*.png)')
parser.add_argument('--dims', metavar='dims', type=str, default='XY',
                help='dimensions of the image (XY,YX,XYC,YXC, default=XY)')
parser.add_argument('--clipping', metavar='clipping', type=str, default='minmax',
                help='clipping approach (imageclip,minmax,zeromax default=minmax) \n \t imageclip: make output image in the same range input. \n \t minmax: apply min max normalization and makes between 0 and 1. \n \t zeromax: clip between 0 and max of input image')
parser.add_argument('--format-out', metavar='format-out', type=str, default='.png',
                help='format of the output. make png makes a RGB image in gray scale if XY(png, .tif default: .png)')
args = parser.parse_args()
print(args)

def create_output_directory(output_path):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.target),'denoised_images')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path
def clip(pred, lb, ub):
    if args.clipping == 'zeromax':
        pred = pred.copy()
        pred[pred<0] = 0
    pred = (pred-pred.min())/(pred.max()-pred.min())
    pred = (ub-lb)*pred + lb
    return pred

def denoise_images(images_path, output_path):
    model_name = 'N2V'
    basedir = 'models'
    model = N2V(config=None, name=model_name, basedir=basedir)
    warning_print = False
    print('DENOISING......')
    for f in tqdm(os.listdir(images_path)):
        if os.path.isfile(os.path.join(images_path,f)) and f.endswith(args.fileName.replace('*','')):
            img = imread(os.path.join(images_path,f))

            # if args.dims = N means Not Defined. Then the program decides which is the one.
            if len(img.shape)==2 and args.dims=='N':
                axes = 'YX'
            elif len(img.shape)==3 and args.dims=='N':
                axes = 'YXC'
            else:
                axes = args.dims
            # Ensures that the input has 3 channels if read image axes are XYC or YXC
            if 'C' in axes and img.shape[-1]==4:
                if not warning_print:
                    print('Warning: alpha channels will be removed from all images.')
                    warning_print = True
                    print('Org. shape: ', img.shape, 'New shape:', img[...,:3].shape)
                img = img[...,:3]
            if len(axes) == 2 and not len(img.shape)==2:
                if not warning_print:
                    print('Warning: Enfoced 2 channels. 3 channels detected. All images will be converted.')
                    warning_print = True
                    print('Org. shape: ', img.shape, 'New shape:', img[...,0].shape)
                img = img[...,0] # taking the red channel
            pred = model.predict(img, axes=axes)
            f_out_d = os.path.join(output_path, 'Denois-' + f.replace(args.fileName.replace('*',''),args.format_out))
            f_out_s = os.path.join(output_path, 'Sample-' + f.replace(args.fileName.replace('*',''),args.format_out))
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
            else:
                raise Exception('Invalid input value clipping not supported.' + args.clipping + '. Check --help for datails.')
            if args.format_out == '.png':
                print('saving file denoised : ', f_out_d)
                imsave(f_out_d, clip(pred, lb, ub), cmap='gray')
                print('saving file input to network: ', f_out_s)
                imsave(f_out_s, clip(img, lb, ub), cmap='gray')
            elif args.format_out == '.tif':
                print('saving file denoised : ', f_out_d)
                tiff.imsave(f_out_d, clip(pred, lb, ub))
                print('saving file input to network: ', f_out_s)
                tiff.imsave(f_out_s, clip(img, lb, ub))
            else:
                raise Exception('Other format not supported' + args.format_out)
# Creating the path of denoised images
output_path = create_output_directory(args.output)
print('Output path is: ', output_path )
if args.train=='y' or not os.path.exists('models/N2V/weights_best.h5'):
    training_args = generate_args(data_path=args.target, fileName=args.fileName, dims=args.dims)
    model, X, X_val = prepare_training_data(training_args)
    history = train_model(model, X, X_val)
# apply on video
denoise_images(args.target, output_path)
