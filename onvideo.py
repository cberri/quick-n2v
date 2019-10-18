import argparse
import os
from vtools.videotovideo import run
from vtools.videotoimage import run as create_training_images
from vtools import generate_args, prepare_training_data, train_model

parser = argparse.ArgumentParser(description='Denoise video with N2V')
parser.add_argument('--target', metavar='target', type=str,
                help='target avi video full path')
parser.add_argument('--output', metavar='output', type=str, default = None,
                help='output avi video full path')
args = parser.parse_args()
print(args)

# training of the model
if not os.path.exists('training_images'):
    create_training_images()

if not os.path.exists('models/N2V/weights_best.h5'):
    training_args = generate_args(data_path='training_images')
    model, X, X_val = prepare_training_data(training_args)
    history = train_model(model, X, X_val)
# apply on video
run(args.target, video_out=args.output)
