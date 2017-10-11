from __future__ import print_function

import h5py
import numpy as np
import os

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Flatten, Dense

from syncnet_params import *

#############################################################
# LOAD TRAINED SYNCNET MODEL
#############################################################


def load_pretrained_syncnet_model(version='v4', mode='lip', verbose=False):

    # version = {'v4', 'v7'}
    if version not in {'v4', 'v7'}:
        print("\n\nERROR: version number not valid! Expected 'v4' or 'v7', got:", version, "\n")
        return

    # mode = {lip, audio, both}
    if mode not in {'lip', 'audio', 'both'}:
        print("\n\nERROR: 'mode' not defined properly! Expected one of {'lip', 'audio', 'both'}, got:", mode, "\n")
        return

    try:

        # Load syncnet model
        syncnet_model = load_syncnet_model(version=version, mode=mode, verbose=verbose)

        if verbose:
            print("Loaded syncnet model", version)

        # Read weights and layer names
        syncnet_weights, syncnet_layer_names, audio_start_idx, lip_start_idx = \
            load_syncnet_weights(version=version, verbose=verbose)

        if verbose:
            print("Loaded syncnet weights.")

        # Set video weights to syncnet_video_model
        set_syncnet_weights_to_syncnet_model(syncnet_model=syncnet_model,
                                             syncnet_weights=syncnet_weights,
                                             syncnet_layer_names=syncnet_layer_names,
                                             mode=mode,
                                             verbose=verbose)

        if verbose:
            print("Set syncnet weights.")

    except ValueError as err:
        print(err)
        return

    except KeyboardInterrupt:
        print("\n\nCtrl+C was pressed!\n")
        return

    return syncnet_model


#############################################################
# LOAD SYNCNET MODEL
#############################################################

def load_syncnet_model(version='v4', mode='lip', verbose=False):
    if mode == 'lip':
        if version == 'v4':
            # Load frontal model
            syncnet_model = syncnet_video_model_v4()
        elif version == 'v7':
            # Load multi-view model
            syncnet_model = syncnet_video_model_v7()
    else:
        raise ValueError("\n\nERROR: " + mode + " mode not implemented yet, sorry...\nOnly 'lip' mode available.")
    return syncnet_model


#############################################################
# LOAD SYNCNET WEIGHTS
#############################################################

def load_syncnet_weights(version='v4', verbose=False):

    if version == 'v4':
        syncnet_weights_file = SYNCNET_WEIGHTS_FILE_V4
    elif version == 'v7':
        syncnet_weights_file = SYNCNET_WEIGHTS_FILE_V7

    if verbose:
        print("Loading syncnet_weights from", syncnet_weights_file)

    if not os.path.isfile(syncnet_weights_file):
        raise ValueError(
            "\n\nERROR: syncnet_weight_file missing!! File: " + syncnet_weights_file + \
            "\nPlease specify correct file name in the syncnet_params.py file and relaunch.\n")

    # Read weights file, with layer names
    with h5py.File(syncnet_weights_file, 'r') as f:
        syncnet_weights = [f[v[0]][:] for v in f['net/params/value']]
        syncnet_layer_names = [[chr(i) for i in  f[n[0]]] \
                               for n in f['net/layers/name']]

    # Find the starting index of audio and video layers
    audio_found = False
    audio_start_idx = 0
    lip_found = False
    lip_start_idx = 0

    # Join the chars of layer names to make them words
    for i in range(len(syncnet_layer_names)):
        syncnet_layer_names[i] = ''.join(syncnet_layer_names[i])

        # Finding audio_start_idx
        if not audio_found and 'audio' in syncnet_layer_names[i]:
            audio_found = True
        elif not audio_found and 'audio' not in syncnet_layer_names[i]:
            if 'conv' in syncnet_layer_names[i]:
                audio_start_idx += 2
            elif 'bn' in syncnet_layer_names[i]:
                audio_start_idx += 3
            elif 'fc' in syncnet_layer_names[i]:
                audio_start_idx += 2

        # Finding lip_start_idx
        if not lip_found and 'lip' in syncnet_layer_names[i]:
            lip_found = True
        elif not lip_found and 'lip' not in syncnet_layer_names[i]:
            if 'conv' in syncnet_layer_names[i]:
                lip_start_idx += 2
            elif 'bn' in syncnet_layer_names[i]:
                lip_start_idx += 3
            elif 'fc' in syncnet_layer_names[i]:
                lip_start_idx += 2

        if verbose:
            print("  ", i, syncnet_layer_names[i])

    if verbose:
        print("  lip_start_idx =", lip_start_idx)

    return syncnet_weights, syncnet_layer_names, audio_start_idx, lip_start_idx


#############################################################
# SET WEGHTS TO MODEL
#############################################################

def set_syncnet_weights_to_syncnet_model(syncnet_model,
                                         syncnet_weights,
                                         syncnet_layer_names,
                                         mode = 'lip',
                                         verbose=False):

    if verbose:
        print("Setting weights to model:")

    # Video syncnet-related weights begin at 35 in syncnet_weights
    if mode == 'lip':
        syncnet_weights_idx = 35

    # Init syncnet_layer_idx, to be incremented only at 'lip' layers
    syncnet_layer_idx = -1

    # Load weights layer-by-layer
    for syncnet_layer_name in syncnet_layer_names:

        # Skip the irrelevant layers
        if mode == 'lip' and 'lip' not in syncnet_layer_name:
            continue
        elif mode == 'audio' and 'audio' not in syncnet_layer_name:
            continue

        # Increment the index on the model
        syncnet_layer_idx += 1

        if verbose:
            if mode == 'lip':
                print("  SyncNet Video Layer", syncnet_layer_idx, ":", syncnet_layer_name)
            elif mode == 'audio':
                print("  SyncNet Audio Layer", syncnet_layer_idx, ":", syncnet_layer_name)
            elif mode == 'both':
                if syncnet_layer_idx < len(syncnet_layer_names)/2:
                    print("  SyncNet Audio Layer", syncnet_layer_idx, ":", syncnet_layer_name)
                else:
                    print("  SyncNet Video Layer", syncnet_layer_idx, ":", syncnet_layer_name)


        # Convolutional layer
        if 'conv' in syncnet_layer_name:
            syncnet_model.layers[syncnet_layer_idx].set_weights(
                [np.transpose(syncnet_weights[syncnet_weights_idx], (2, 3, 1, 0)),
                 np.squeeze(syncnet_weights[syncnet_weights_idx + 1])])
            syncnet_weights_idx += 2

        # Batch Normalization layer
        elif 'bn' in syncnet_layer_name:
            syncnet_model.layers[syncnet_layer_idx].set_weights(
                [np.squeeze(syncnet_weights[syncnet_weights_idx]),
                 np.squeeze(syncnet_weights[syncnet_weights_idx + 1]),
                 syncnet_weights[syncnet_weights_idx + 2][0],
                 syncnet_weights[syncnet_weights_idx + 2][1]])
            syncnet_weights_idx += 3

        # ReLU layer
        elif 'relu' in syncnet_layer_name:
            continue

        # Pooling layer
        elif 'pool' in syncnet_layer_name:
            continue

        # Dense (fc) layer
        elif 'fc' in syncnet_layer_name:
            # Skip Flatten layer
            if 'flatten' in syncnet_model.layers[syncnet_layer_idx].name:
                syncnet_layer_idx += 1
            # Set weight to Dense layer
            syncnet_model.layers[syncnet_layer_idx].set_weights(
                [np.reshape(
                    np.transpose(syncnet_weights[syncnet_weights_idx],
                        (2, 3, 1, 0)),
                    (syncnet_weights[syncnet_weights_idx].shape[2]*\
                     syncnet_weights[syncnet_weights_idx].shape[3]*\
                     syncnet_weights[syncnet_weights_idx].shape[1],
                     syncnet_weights[syncnet_weights_idx].shape[0])),
                np.squeeze(syncnet_weights[syncnet_weights_idx + 1])])
            syncnet_weights_idx += 2


#############################################################
# SYNCNET_v4 VIDEO (frontal)
#############################################################


def syncnet_video_model_v4():

    # Image data format
    K.set_image_data_format(IMAGE_DATA_FORMAT)
    if IMAGE_DATA_FORMAT == 'channels_first':
        input_shape = (SYNCNET_CHANNELS, MOUTH_H, MOUTH_W)
    elif IMAGE_DATA_FORMAT == 'channels_last':
        input_shape = (MOUTH_H, MOUTH_W, SYNCNET_CHANNELS)

    model_v4 = Sequential()     # (None, 112, 112, 5)

    # conv1_lip
    model_v4.add(Conv2D(96, (3, 3), padding='valid', name='conv1_lip',
        input_shape=input_shape))  # (None, 110, 110, 96)

    # bin1_lip
    model_v4.add(BatchNormalization(name='bin1_lip'))

    # relu1_lip
    model_v4.add(Activation('relu', name='relu1_lip'))

    # pool1_lip
    model_v4.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1_lip'))   # (None, 54, 54, 96)

    # conv2_lip
    model_v4.add(Conv2D(256, (5, 5), padding='valid', name='conv2_lip'))   # (None, 256, 50, 50)

    # bn2_lip
    model_v4.add(BatchNormalization(name='bn2_lip'))

    # relu2_lip
    model_v4.add(Activation('relu', name='relu2_lip'))

    # pool2_lip
    model_v4.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2_lip'))   # (None, 24, 24, 256)

    # conv3_lip
    model_v4.add(Conv2D(512, (3, 3), padding='valid', name='conv3_lip'))   # (None, 22, 22, 512)

    # bn3_lip
    model_v4.add(BatchNormalization(name='bn3_lip'))

    # relu3_lip
    model_v4.add(Activation('relu', name='relu3_lip'))

    # conv4_lip
    model_v4.add(Conv2D(512, (3, 3), padding='valid', name='conv4_lip'))   # (None, 20, 20, 512)

    # bn4_lip
    model_v4.add(BatchNormalization(name='bn4_lip'))

    # relu4_lip
    model_v4.add(Activation('relu', name='relu4_lip'))

    # conv5_lip
    model_v4.add(Conv2D(512, (3, 3), padding='valid', name='conv5_lip'))   # (None, 18, 18, 512)

    # bn5_lip
    model_v4.add(BatchNormalization(name='bn5_lip'))

    # relu5_lip
    model_v4.add(Activation('relu', name='relu5_lip'))

    # pool5_lip
    model_v4.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid', name='pool5_lip'))   # (None, 6, 6, 512)

    # fc6_lip
    model_v4.add(Flatten(name='flatten'))
    model_v4.add(Dense(256, name='fc6_lip'))    # (None, 256)

    # bn6_lip
    model_v4.add(BatchNormalization(name='bn6_lip'))

    # relu6_lip
    model_v4.add(Activation('relu', name='relu6_lip'))

    # fc7_lip
    model_v4.add(Dense(128, name='fc7_lip'))    # (None, 128)

    # bn7_lip
    model_v4.add(BatchNormalization(name='bn7_lip'))

    # relu7_lip
    model_v4.add(Activation('relu', name='relu7_lip'))

    return model_v4

#############################################################
# SYNCNET_v7 VIDEO (multi-view)
#############################################################


def syncnet_video_model_v7():

    # Image data format
    K.set_image_data_format(IMAGE_DATA_FORMAT)
    if IMAGE_DATA_FORMAT == 'channels_first':
        input_shape = (SYNCNET_CHANNELS, FACE_H, FACE_W)
    elif IMAGE_DATA_FORMAT == 'channels_last':
        input_shape = (FACE_H, FACE_W, SYNCNET_CHANNELS)

    model_v7 = Sequential()     # (None, 224, 224, 5)

    # conv1_lip
    model_v7.add(Conv2D(96, (7, 7), strides=(2, 2), padding='valid', name='conv1_lip',
        input_shape=input_shape))    # (None, 109, 109, 96)

    # bin1_lip
    model_v7.add(BatchNormalization(name='bin1_lip'))

    # relu1_lip
    model_v7.add(Activation('relu', name='relu1_lip'))

    # pool1_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool1_lip'))   # (None, 54, 54, 96)

    # conv2_lip
    model_v7.add(Conv2D(256, (5, 5), strides=(2, 2), padding='valid', name='conv2_lip'))   # (None, 25, 25, 96)

    # bn2_lip
    model_v7.add(BatchNormalization(name='bn2_lip'))

    # relu2_lip
    model_v7.add(Activation('relu', name='relu2_lip'))

    # pool2_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2_lip'))   # (None, 12, 12, 256)

    # conv3_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same', name='conv3_lip'))   # (None, 12, 12, 512)

    # bn3_lip
    model_v7.add(BatchNormalization(name='bn3_lip'))

    # relu3_lip
    model_v7.add(Activation('relu', name='relu3_lip'))

    # conv4_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same', name='conv4_lip'))   # (None, 12, 12, 512)

    # bn4_lip
    model_v7.add(BatchNormalization(name='bn4_lip'))

    # relu4_lip
    model_v7.add(Activation('relu', name='relu4_lip'))

    # conv5_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same', name='conv5_lip'))   # (None, 12, 12, 512)

    # bn5_lip
    model_v7.add(BatchNormalization(name='bn5_lip'))

    # relu5_lip
    model_v7.add(Activation('relu', name='relu5_lip'))

    # pool5_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5_lip'))   # (None, 6, 6, 256)

    # fc6_lip
    model_v7.add(Flatten(name='flatten'))
    model_v7.add(Dense(512, name='fc6_lip'))

    # bn6_lip
    model_v7.add(BatchNormalization(name='bn6_lip'))

    # relu6_lip
    model_v7.add(Activation('relu', name='relu6_lip'))

    # fc7_lip
    model_v7.add(Dense(256, name='fc7_lip'))

    # bn7_lip
    model_v7.add(BatchNormalization(name='bn7_lip'))

    # relu7_lip
    model_v7.add(Activation('relu', name='relu7_lip'))

    return model_v7
