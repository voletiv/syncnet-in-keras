# SYNCNET - PARAMS

from __future__ import print_function

import h5py
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Flatten, Dense

# Set channels_last
K.set_image_data_format('channels_last')

#############################################################
# PARAMS
#############################################################

SYNCNET_WEIGHTS_FILE_V4 = 'syncnet-weights/lipsync_v4_73.mat'

SYNCNET_WEIGHTS_FILE_V7 = 'syncnet-weights/lipsync_v7_73.mat'

#############################################################
# CONSTANTS
#############################################################

MOUTH_H = 112

MOUTH_W = 112

FACE_H = 224

FACE_W = 224

SYNCNET_CHANNELS = 5

