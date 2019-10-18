import argparse
import os

from tqdm import tqdm
from vtools import generate_args, prepare_training_data, train_model
from matplotlib.image import imread, imsave
from n2v.models import N2V
import numpy as np


parser = argparse.ArgumentParser(description='Denoise video with N2V')
parser.add_argument('--target', metavar='target', type=str,
                help='target directory containing png images full path')
parser.add_argument('--output', metavar='output', type=str, default = None,
                help='output directory full path')
args = parser.parse_args()
print(args)

def create_output_directory(output_path):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.target),'denoised_images')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path
def denoise_images(images_path, output_path):
    model_name = 'N2V'
    basedir = 'models'
    model = N2V(config=None, name=model_name, basedir=basedir)
    for f in tqdm(os.listdir(images_path)):
        if os.path.isfile(os.path.join(images_path,f)) and f.endswith('.png'):
            img = imread(os.path.join(images_path,f))
            if len(img.shape)==2:
                axes = 'YX'
            elif len(img.shape)==3:
                axes = 'YXC'
            else:
                print('Invalid format image: ', img.shape, 'formats supported YX and YXC')
                break
            pred = model.predict(img, axes=axes)
            f_out = os.path.join(output_path, f)
            imsave(f_out, np.clip(pred, 0.0, 1.0), cmap='gray')

# Creating the path of denoised images
output_path = create_output_directory(args.output)

if not os.path.exists('models/N2V/weights_best.h5'):
    training_args = generate_args(data_path='video_images')
    model, X, X_val = prepare_training_data(training_args)
    history = train_model(model, X, X_val)
# apply on video
denoise_images(args.target, output_path)
