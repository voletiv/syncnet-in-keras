from syncnet_params import *

#############################################################
# LOAD TRAINED SYNCNET MODEL
#############################################################


def load_pretrained_syncnet_video_model(version='v4', verbose=False):

    # Load syncnet model
    if version == 'v4':
        # Load frontal model
        syncnet_model = syncnet_video_model_v4()
        syncnet_weights_file = SYNCNET_WEIGHTS_FILE_V4
    elif version == 'v7':
        # Load multi-view model
        syncnet_model = syncnet_video_model_v7()
        syncnet_weights_file = SYNCNET_WEIGHTS_FILE_V7
    else:
        print("\n\nERROR: version number not valid! Expected 'v4' or 'v7', got:",
            version, "\n")
        return

    if verbose:
        print("Loaded syncnet model", version)

    # Read weights and layer names
    syncnet_weights, syncnet_layer_names = load_syncnet_weights(syncnet_weights_file)

    if verbose:
        print("Loaded syncnet weights from", syncnet_weights_file)

    # Video syncnet-related weights begin at 35 in syncnet_weights
    syncnet_weights_idx = 35

    # Init syncnet_layer_idx, to be incremented only at 'lip' layers
    syncnet_layer_idx = -1

    # Load weights layer-by-layer
    for syncnet_layer_name in syncnet_layer_names:

        # For the video layers
        if 'lip' in syncnet_layer_name:

            syncnet_layer_idx += 1

            if verbose:
                print("SyncNet Video Layer", syncnet_layer_idx, ":", syncnet_layer_name)

            # Convolutional layer
            if 'conv' in syncnet_layer_name:
                syncnet_model.layers[syncnet_layer_idx].set_weights([np.transpose(syncnet_weights[syncnet_weights_idx], (2, 3, 1, 0)),
                    np.squeeze(syncnet_weights[syncnet_weights_idx + 1])])
                syncnet_weights_idx += 2

            # Batch Normalization layer
            elif 'bn' in syncnet_layer_name:
                syncnet_model.layers[syncnet_layer_idx].set_weights([np.squeeze(syncnet_weights[syncnet_weights_idx]),
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
                syncnet_model.layers[syncnet_layer_idx].set_weights([np.reshape(
                        np.transpose(syncnet_weights[syncnet_weights_idx],
                            (2, 3, 1, 0)), (
                        syncnet_weights[syncnet_weights_idx].shape[2]*\
                        syncnet_weights[syncnet_weights_idx].shape[3]*\
                        syncnet_weights[syncnet_weights_idx].shape[1],
                        syncnet_weights[syncnet_weights_idx].shape[0])),
                    np.squeeze(syncnet_weights[syncnet_weights_idx + 1])])
                syncnet_weights_idx += 2

    return syncnet_model


#############################################################
# LOAD SYNCNET WEGHTS (frontal)
#############################################################

def load_syncnet_weights(syncnet_weights_file):

    # Read weights file, with layer names
    with h5py.File(syncnet_weights_file, 'r') as f:
        syncnet_weights = [f[n[0]][:] for n in f['net/params/value']]
        syncnet_layer_names = [[chr(i) for i in  f[n[0]]] for n in f['net/layers/name']]

    # Join the chars of layer names to make them words
    for i in range(len(syncnet_layer_names)):
        syncnet_layer_names[i] = ''.join(syncnet_layer_names[i])

    return syncnet_weights, syncnet_layer_names


#############################################################
# SYNCNET_v4 (frontal)
#############################################################


def syncnet_video_model_v4():

    model_v4 = Sequential()     # (None, 112, 112, 5)

    # conv1_lip
    model_v4.add(Conv2D(96, (3, 3), padding='valid', name='conv1_lip',
        input_shape=(MOUTH_H, MOUTH_W, SYNCNET_CHANNELS)))  # (None, 110, 110, 96)

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
# SYNCNET_v7 (multi-view)
#############################################################


def syncnet_model_v7():

    model_v7 = Sequential()     # (None, 224, 224, 5)

    # conv1_lip
    model_v7.add(Conv2D(96, (7, 7), strides=(2, 2), padding='valid', input_shape=(FACE_H, FACE_W, SYNCNET_CHANNELS)))    # (None, 109, 109, 96)

    # bin1_lip
    model_v7.add(BatchNormalization())

    # relu1_lip
    model_v7.add(Activation('relu'))

    # pool1_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))   # (None, 54, 54, 96)

    # conv2_lip
    model_v7.add(Conv2D(256, (5, 5), strides=(2, 2), padding='valid'))   # (None, 25, 25, 96)

    # bn2_lip
    model_v7.add(BatchNormalization())

    # relu2_lip
    model_v7.add(Activation('relu'))

    # pool2_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))   # (None, 12, 12, 256)

    # conv3_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same'))   # (None, 12, 12, 512)

    # bn3_lip
    model_v7.add(BatchNormalization())

    # relu3_lip
    model_v7.add(Activation('relu'))

    # conv4_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same'))   # (None, 12, 12, 512)

    # bn4_lip
    model_v7.add(BatchNormalization())

    # relu4_lip
    model_v7.add(Activation('relu'))

    # conv5_lip
    model_v7.add(Conv2D(512, (3, 3), padding='same'))   # (None, 12, 12, 512)

    # bn5_lip
    model_v7.add(BatchNormalization())

    # relu5_lip
    model_v7.add(Activation('relu'))

    # pool5_lip
    model_v7.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))   # (None, 6, 6, 256)

    # fc6_lip
    model_v4.add(Flatten())
    model_v4.add(Dense(512))

    # bn6_lip
    model_v4.add(BatchNormalization())

    # relu6_lip
    model_v4.add(Activation('relu'))

    # fc7_lip
    model_v4.add(Dense(256))

    # bn7_lip
    model_v4.add(BatchNormalization())

    # relu7_lip
    model_v4.add(Activation('relu'))







#############################################################
# PRINT STRUCTURE
#############################################################


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.
    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))
        if len(f.items())==0:
            return
        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))
            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param))
    finally:
        f.close()



