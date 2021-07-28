## quick-n2v
Quick scripts to run noise2void tool on videos or images

## Installation
```
$ conda create -n n2v pip python==3.7
$ conda install tensorflow or tensorflow-gpu 
$ pip install -r requirements.txt
```

Running the scripts

For videos (avi)

```
$ python onvideo.py --target FULL_PATH_TO_VIDEO_AVI
``` 

For images (png)
```
$ python onimages.py --target FULL_PATH_TO_IMAGES
```

For more parameters (png)
```
$ python onimagestunning.py --target FULL_PATH_TO_IMAGES
```

CUDA 10.1 drivers installation Ubuntu 18.04 LTS
(https://medium.com/@exesse/cuda-10-1-installation-on-ubuntu-18-04-lts-d04f89287130)


### OPTIONAL: Remove previous nvidia drivers
sudo rm /etc/apt/sources.list.d/cuda*

sudo apt remove --autoremove nvidia-cuda-toolkit

sudo apt remove --autoremove nvidia-*

### Setup the correct CUDA PPA on your system
```
sudo apt update

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb 
http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

### Install CUDA 10.1 packages
```
sudo apt update

sudo apt install cuda-10-1

sudo apt install libcudnn7
```

### Specify PATH to CUDA in ~/.profile file (use subline text)
```
sudo subl ~/.profile
```

### Set PATH for cuda 10.1 installation
Edit ~/.bashrc
```
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```
### Restart ubuntu
```
sudo reboot
```

### Verify installation
```
nvcc -V

nvidia-smi
```
### OPTIONAL TOOLS (https://github.com/wookayin/gpustat)
```
pip install gpustat

gpustat –watch
Install Anaconda3 & Quick Nosie 2 Void Master
(https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
```
### Install curl
```
sudo apt update 

sudo apt upgrade

sudo apt install curl
```
### Download anaconda in your temporary directory
```
cd /tmp

curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
```
### Verify installer [OPTIONAL]
```
sha256sum Anaconda3-2019.10-Linux-x86_64.sh

	69c64167b8cf3a8fc6b50d12d8476337
```
### Install anaconda
```
bash Anaconda3-2019.10-Linux-x86_64.sh
```

### Activate the installation
``` 
source ~/.bashrc
```

### Verify installation
```
conda list
```
## Nosie 2 Void
### Create conda environment
```
conda create -n n2v pip python==3.7
```
### Activate the n2v conda environment
``` 
conda activate n2v
```
### In case of tensorflow GPU use tensorflow-gpu=1.14
```
(n2v) >> conda install tensorflow=1.14 keras=2.2.4
```
### Install jupyter [OPTIONAL]
```
(n2v) >> pip install jupyter
```
### Download quick-n2v-master from 
https://github.com/aiporre/quick-n2v

### Navigate to quick-n2v-master directory
```
(n2v) >> sudo apt install git

(n2v) >> pip install -r requirements.txt
```
### Look for help functions
```
(n2v) >> python onimages.py --help

Denoise video with N2V

optional arguments:
  -h, --help            show this help message and exit
  --target target       target directory containing png images full path
  --output output       output directory full path
  --train train         force train? y or n (default=n)
  --fileName fileName   file name ending (default=*.tif)
  --dims dims           dimensions of the image (XY,YX,XYC,YXC, default=XY)
  --clipping clipping   clipping approach (imageclip,minmax,zeromax, 0255,
                        default=minmax) imageclip: make output image in the
                        same range input. minmax: apply min max normalization
                        and makes between 0 and 1. zeromax: clip between 0 and
                        max of input image. 0255: means clips prediction from
                        0 to 255
  --formatOut formatOut
                        format of the output. Noticed that when png and XY it
                        makes a RGB image in gray scale (png, .tif default:
                        .tif)
  --saveInputs          save inputs to the network that maybe have been
                        converted (y, n default: n)
  --stack               save inputs to the network that maybe have been
                        converted (y, n default: n)


Run quick-n2v-master with a sample image (test on CPU)
```
### OPTIONAL, in case of multiples GPUs (add the following to the first line of onimages.py to specify the GPU for the task)
```
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" ### 0, 1, 2, 3
```

### Run quick-n2v-master
```
(n2v) >> python onimages.py --target ./n2v_input/2019-11-29 --dims YX --train y

 ~/tf.py:275: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.
 ....

Epoch 1/100
 60/400 [===>..........................] - ETA: 7:24 - loss: 0.9071 - n2v_mse: 0.9071 - n2v_abs: 0.7417^CTraceback (most recent call last):
 ...
```
### Lets run it!



