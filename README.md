# quick-n2v
Quick scripts to run noise2void tool on videos or images

Installation
```
$ conda create -n n2v pip python==3.7
$ conda install tensorflow or tensorflow-gpu 
$ pip install -r requirements.txt
```

Running the scripts

For videos (avi)

```
$ python onvideos.py --target FULL_PATH_TO_VIDEO_AVI
``` 

For images (png)
```
$ python onvideos.py --target FULL_PATH_TO_DIRECTORY
```

