## Installation Steps quick-n2v for TF >= 2.0 CUDA 11.0

Source: https://github.com/juglab/n2v

1. Create the n2v conda environment

   `$ conda create -n n2v pip python==3.7`

   

2. Activate the n2v conda environment

   `$ conda activate n2v`

   

3. Install tensorflow and keras

   `$ pip install tensorflow-gpu==2.4.0 keras==2.3.1`

   

   Test tensorflow installation

   ```python
   $ python
   $ import tensorflow as tf
   $ print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
   
   $ Num GPUs Available: X
   $ exit()
   ```

   

4. Install jupyter notebook

   `$ pip install jupyter`

   

5. Clone the quick-n2v repository

   `$ https://github.com/cberri/quick-n2v.git`

   

6. Install the n2v dependency

   `$ pip install -r requirements.txt`

   

7. Install the last n2v release

   `$ pip install n2v`

   If you want to install a different version of n2v specify the version number. The following works for TF 1.x.x

   `$ pip install n2v==0.2.1`

   

8. Run the n2v help

   `$ python n2v_Modified_2020-09-21_TF2.py --help`

   ```python
   optional arguments:
     -h, --help            show this help message and exit
     --target target       path to the target directory containing the input
                           images
     --output output       output directory full path
     --baseDir baseDir     directory path to store the trained models and the
                           configurations.
     --train train         force train? y or n (default=n)
     --dims dims           dimensions of your data, can include: X,Y,Z,C
                           (default=YX)
     --fileName fileName   file name ending (default=*.tif)
     --clipping clipping   clipping approach (imageclip,minmax,zeromax,0255
                           default=minmax) imageclip: make output image in the
                           same range input. minmax: apply min max normalization
                           and makes between 0 and 1. zeromax: clip between 0 and
                           max of input image, 0255: clip the prediction between
                           0 and 255.
     --formatOut formatOut
                           format of the output. Noticed that when png and XY it
                           makes a RGB image in gray scale (png, .tif default:
                           .tif)
     --saveInputs          save inputs to the network that maybe have been
                           converted (y, n default: n)
     --stack               process images as stack of 2D images (y, n default: n)
     --name name           name of your network default=N2V
     --validationFraction VALIDATIONFRACTION
                           Fraction of data you want to use for validation
                           (percent default=10.0)
     --patchSizeXY         XY-size of your training patches (default=64)
     --patchSizeZ PATCHSIZEZ
                           Z-size of your training patches (default=64)
     --epochs EPOCHS       number of training epochs (default=100)
     --stepsPerEpoch STEPSPEREPOCH
                           number training steps per epoch (default=400)
     --batchSize BATCHSIZE
                           size of your training batches (default=64)
     --netDepth NETDEPTH   depth of your U-Net (default=2)
     --netKernelSize NETKERNELSIZE
                           Size of conv. kernels in first layer (default=3)
     --n2vPercPix N2VPERCPIX
                           percentage of pixels to manipulated by N2V
                           (default=1.6)
     --learningRate LEARNINGRATE
                           initial learning rate (default=0.0004)
     --unet_n_first UNET_N_FIRST
                           number of feature channels in the first u-net layer
                           (default=32)
     --gpu GPU             default gpu is 0
   ```

   

   run the n2v training (2D example)

   

   ```python
   $ python n2v_Modified_2020-09-21_TF2.py --target "C:\PATH_TO_IMAGES" --baseDir "C:\PATH_TO_MODEL_DIR" --dims YX --train y --gpu 0
   ```

    

   run the n2v prediction (2D example)

   

   ```python
   $ python n2v_Modified_2020-09-21_TF2.py --target "C:\PATH_TO_IMAGES" --baseDir "C:\PATH_TO_MODEL_DIR" --dims YX --train n --gpu 0
   ```
