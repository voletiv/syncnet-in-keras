# syncnet-in-keras

Keras version of SyncNet, by Joon Son Chung and Andrew Zisserman.

SyncNet paper: ["Out of time: automated lip sync in the wild"](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf)

VGG webpage: [VGG - SyncNet](http://www.robots.ox.ac.uk/~vgg/software/lipsync/)

## Requirements

1. Libraries required by Python (I used Python 3) are mentioned in the _requirements.txt_ file.

2. [Keras](https://keras.io/#installation)

3. Pre-trained weights, to be placed in the _syncnet-weights_ directory. Instructions to download and place the files are available in the readme file inside the _syncnet-weights_ directory

## IMPORTANT

- SyncNet takes input images of size (112, 112, 5).

- These input images have pixel values between 0 and 255! **DON'T** rescale image values to [0, 1], keep them in [0, 255]. 

