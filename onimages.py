import argparse
import datetime
import os
import tifffile as tiff
from tqdm import tqdm
from vtools import generate_args, prepare_training_data, train_model
from matplotlib.image import imread, imsave
from n2v.models import N2V
import numpy as np
import tempfile
import re
import shutil

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
                help='clipping approach (imageclip,minmax,zeromax default=minmax) \n \t imageclip: make output image in the same range input.  minmax: apply min max normalization and makes between 0 and 1. zeromax: clip between 0 and max of input image. 0255: means clips prediction from 0 to 255')
parser.add_argument('--formatOut', metavar='formatOut', type=str, default='.png',
                help='format of the output. Noticed that when png and XY it makes a RGB image in gray scale (png, .tif default: .png)')
parser.add_argument('--saveInputs', metavar='', type=str, default='n',
                help='save inputs to the network that maybe have been converted (y, n default: n)')
parser.add_argument('--stack', metavar='', type=str, default='n',
                help='save inputs to the network that maybe have been converted (y, n default: n)')
args = parser.parse_args()
print(args)
if args.stack == 'y':
    assert args.fileName.endswith('.tif') and args.formatOut.endswith('.tif'), 'Stacked image input is "activated". Therefore, formatOut=.tif and fileName=*.tif must be selected'

def create_output_directory(output_path):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.target),'denoised_images'+str(datetime.datetime.now()).replace(' ','_').replace('.','p').replace(':','T'))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path

def clip(pred, lb, ub):
    if args.clipping == 'zeromax' or args.clipping == '0255':
        pred = pred.copy()
        pred[pred<0] = 0
    if args.clipping == '0255':
        pred = pred.copy()
        pred[pred>255] = 255
        # avoids normalization
        return pred

    pred = (pred-pred.min())/(pred.max()-pred.min())
    pred = (ub-lb)*pred + lb
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
        if os.path.isfile(os.path.join(images_path,f)) and f.endswith('.tif'):
            img = tiff.imread(os.path.join(images_path,f))
            for ii, im in enumerate(img):
                f_out = os.path.join(temp_dir, f.split('.tif')[0] + '.' +str(ii) + '.tif')
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
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

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

        if os.path.isfile(os.path.join(images_path,f)) and f.endswith('.tif'):
            all_files_relevant.append(f)
    sort_nicely(all_files_relevant)
    # create stacks once the images are sorted
    n0 = -1
    image_paths = []
    print('Concatenating images...')
    N = len(all_files_relevant)-1
    for i, f in tqdm(enumerate(all_files_relevant)):
        n1 = int(f.split('.')[-2])
        if n1<n0 or i==N:
            # generate stacked image
            file_name_stack = os.path.basename(image_paths[0]).split('.0.tif')[0] + '.tif'
            f_out = os.path.join(output_path, file_name_stack )
            imgs = np.array([tiff.imread(img_path) for img_path in image_paths])
            tiff.imsave(f_out,imgs)
            n0 = -1
            image_paths = []
        else:
            # sequence continues
            image_paths.append(os.path.join(images_path,f))
            n0 = n1

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
            if args.saveInputs=='y':
                f_out_d = os.path.join(output_path, 'Denoised-' + f.replace(args.fileName.replace('*',''),args.formatOut))
                f_out_s = os.path.join(output_path, 'Input-' + f.replace(args.fileName.replace('*',''),args.formatOut))
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
                raise Exception('Invalid input value clipping not supported.' + args.clipping + '. Check --help for datails.')
            if args.formatOut == '.png':
                print('saving file denoised : ', f_out_d)
                imsave(f_out_d, clip(pred, lb, ub), cmap='gray')
                if args.saveInputs=='y':
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
print('Output path is: ', output_path )
if args.stack == 'y':
    print('WARNING: Stack input selected. temporary files will be generated')
    unravel_path = create_unravel_folder(args.target)
    unravel_output_path = create_output_directory(os.path.join(unravel_path,'unraveled_denoised'))
    if args.train=='y' or not os.path.exists('models/N2V/weights_best.h5'):
        training_args = generate_args(data_path=unravel_path, fileName=args.fileName, dims=args.dims)
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
                                      baseDir=args.baseDir, name=args.Name, validationFraction=args.validationFraction,
                                      patchSizeXY=args.patchSizeXY, patchSizeZ=args.patchSizeZ, epochs=args.epochs,
                                      stepsPerEpoch=args.stepsPerEpoch, batchSize=args.batchSize, netDepth=args.netDepth,
                                      netKernelSize=args.netKernelSize, n2vPercPix=args.n2vPercPix,
                                      learningRate=args.learningRate, unet_n_first=args.unet_n_first)
        model, X, X_val = prepare_training_data(training_args)
        history = train_model(model, X, X_val)
    # apply on video
    denoise_images(args.target, output_path)
