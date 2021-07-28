import argparse
import os
from vtools.videotovideo import run
from vtools.videotoimage import run as create_training_images
from vtools import generate_args, prepare_training_data, train_model

parser = argparse.ArgumentParser(description='Denoise video with N2V')
parser.add_argument('--target', metavar='target', type=str,
                help='target avi video full path')
parser.add_argument('--clipping', metavar='clipping', type=str, default='minmax',
                help='clipping approach (imageclip,minmax,zeromax,  0255, default=minmax) \n \t imageclip: make output image in the same range input.  minmax: apply min max normalization and makes between 0 and 1. zeromax: clip between 0 and max of input image. 0255: means clips prediction from 0 to 255')
parser.add_argument('--gray', metavar='gray', type=str, default='y',
                help='make gray before predict')
parser.add_argument('--output', metavar='output', type=str, default = None,
                help='output avi video full path')
args = parser.parse_args()
print(args)

# training of the model
if not os.path.exists('video_images'):
    os.mkdir('video_images')
    create_training_images(args.target, output_path='video_images', gray=args.gray)

if not os.path.exists('models/N2V/weights_best.h5'):
    training_args = generate_args(data_path='video_images')
    model, X, X_val = prepare_training_data(training_args)
    history = train_model(model, X, X_val)
# apply on video
class Clipping:
    def __init__(self, method):
        self.method = method
    def __call__(self, pred, *args, **kwargs):

        if self.method == 'imageclip':
            ub = pred.max()
            lb = pred.min()
        elif self.method == 'zeromax':
            ub = pred.max()
            lb = 0
        elif self.method == 'minmax':
            ub = 1
            lb = 0
        elif self.method == '0255':
            ub = 1
            lb = 0
        else:
            raise Exception(
                'Invalid input value clipping not supported.' + self.method + '. Check --help for datails.')
        if self.method == 'zeromax' or self.method == '0255':
            pred = pred.copy()
            pred[pred < 0] = 0
        if self.method == '0255':
            pred = pred.copy()
            pred[pred > 255] = 255
            # avoids normalization
            return pred
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred = (ub - lb) * pred + lb
        return pred
clipping = Clipping(args.clipping)
run(args.target, video_out=args.output, gray=args.gray, custom_transform=clipping)
