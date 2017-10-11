# syncnet-weights

Place pre-trained SyncNet weight files here.

## [2017-10-11]

Currently, pre-trained weights are available on the [VGG webpage](http://www.robots.ox.ac.uk/~vgg/software/lipsync/) for SyncNet. The weight files are:

- "syncnet_v4.mat"

- "syncnet_v7.mat"

"v4" corresponds to model trained on frontal faces, while "v7" is the one trained on multi-view faces.

But these weights are in .mat format, and they have not been saved via Matlab 7.3 version or later. Importing them into python is difficult. Weights saved via Matlab 7.3 version are required.

Load these into Matlab 7.3 and save them to be able to use them in Keras.

Or contact me for the weights, I'm available at vikram.voleti@gmail.com.
