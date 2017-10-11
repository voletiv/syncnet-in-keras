# SYNCNET - LOAD WEIGHTS FROM .mat FILE

import h5py
import numpy as np

modelFile_v4 = '/home/voletiv/TRAINED-MODELS/syncnet/lipsync_v4_73.mat'
modelFile_v7 = '/home/voletiv/TRAINED-MODELS/syncnet/lipsync_v7_73.mat'

# Read weights file
with h5py.File(modelFile_v4, 'r') as f:
    weights_v4 = [f[n[0]][:] for n in f['net/params/value']]
    names_v4 = [[chr(i) for i in  f[n[0]]] for n in f['net/layers/name']]

with h5py.File(modelFile_v7, 'r') as f:
    weights_v7 = [f[n[0]][:] for n in f['net/params/value']]
    names_v7 = [[chr(i) for i in  f[n[0]]] for n in f['net/layers/name']]

for i in range(len(names_v4)):
    names_v4[i] = ''.join(names_v4[i])

for i in range(len(names_v7)):
    names_v7[i] = ''.join(names_v7[i])

# Print layer shapes
for l in weights_v4:
    print(l.shape)

for l in weights_v7:
    print(l.shape)

# Print layer names
names_v7

# Video weights start at 35

model.layers[0].set_weights([np.transpose(weights[35], (3, 2, 1, 0)), np.squeeze(weights[36])])

# Read .mat file

with h5py.File(modelFile, 'r') as file:
    print(list(file.keys()))

# ['#refs#', 'net']

file = h5py.File(modelFile, 'r')

[n for n in file['net']]
# ['layers', 'meta', 'params', 'vars']


[n for n in file['net/layers']]
# ['block', 'inputs', 'name', 'outputs', 'params', 'type']

[n for n in file['net/meta']]
# ['augmentation_a', 'augmentation_l', 'normalization_a', 'normalization_l', 'trainOpts']

[n for n in file['net/params']]
# ['learningRate', 'name', 'value', 'weightDecay']

[n for n in file['net/vars']]
# ['name', 'precious']



# Layer names

with h5py.File(modelFilev7, 'r') as f:
    names_v7 = [[chr(i) for i in  file[n[0]]] for n in file['net/layers/name']]

for i in range(len(names_v7)):
    names_v7[i] = ''.join(names_v7[i])

# >>> names
# ['conv1_audio', 'bn1_audio', 'relu1_audio', 'conv2_audio', 'bn2_audio', 'relu2_audio', 'pool2_audio', 'conv3_audio', 'bn3_audio', 'relu3_audio', 'conv4_audio', 'bn4_audio', 'relu4_audio', 'conv5_audio', 'bn5_audio', 'relu5_audio', 'pool5_audio', 'fc6_audio', 'bn6_audio', 'relu6_audio', 'fc7_audio', 'bn7_audio', 'relu7_audio',
# 'conv1_lip', 'bn1_lip', 'relu1_lip', 'pool1_lip', 'conv2_lip', 'bn2_lip', 'relu2_lip', 'pool2_lip', 'conv3_lip', 'bn3_lip', 'relu3_lip', 'conv4_lip', 'bn4_lip', 'relu4_lip', 'conv5_lip', 'bn5_lip', 'relu5_lip', 'pool5_lip', 'fc6_lip', 'bn6_lip', 'relu6_lip', 'fc7_lip', 'bn7_lip', 'relu7_lip', 'dist', 'loss']


[n for n in file['net/layers']]

# v4
[b.shape for b in weights_v4]
[(64, 1, 3, 3), (1, 64), (1, 64), (1, 64), (2, 64), (128, 64, 3, 3), (1, 128), (1, 128), (1, 128), (2, 128), (256, 128, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (256, 256, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (256, 256, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (256, 256, 4, 5), (1, 256), (1, 256), (1, 256), (2, 256), (128, 256, 1, 1), (1, 128), (1, 128), (1, 128), (2, 128),
(96, 5, 3, 3), (1, 96), (1, 96), (1, 96), (2, 96), (256, 96, 5, 5), (1, 256), (1, 256), (1, 256), (2, 256), (512, 256, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (512, 512, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (512, 512, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (256, 512, 6, 6), (1, 256), (1, 256), (1, 256), (2, 256), (128, 256, 1, 1), (1, 128), (1, 128), (1, 128), (2, 128)]

# v7
[b.shape for b in weights_v7]
[(64, 1, 3, 3), (1, 64), (1, 64), (1, 64), (2, 64), (128, 64, 3, 3), (1, 128), (1, 128), (1, 128), (2, 128), (256, 128, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (256, 256, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (256, 256, 3, 3), (1, 256), (1, 256), (1, 256), (2, 256), (512, 256, 4, 5), (1, 512), (1, 512), (1, 512), (2, 512), (256, 512, 1, 1), (1, 256), (1, 256), (1, 256), (2, 256),
(96, 5, 7, 7), (1, 96), (1, 96), (1, 96), (2, 96), (256, 96, 5, 5), (1, 256), (1, 256), (1, 256), (2, 256), (512, 256, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (512, 512, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (512, 512, 3, 3), (1, 512), (1, 512), (1, 512), (2, 512), (512, 512, 6, 6), (1, 512), (1, 512), (1, 512), (2, 512), (256, 512, 1, 1), (1, 256), (1, 256), (1, 256), (2, 256)]


learningRate = np.squeeze([file[e[0]][:] for e in file['net/params/learningRate']])
name = np.squeeze([file[e[0]][:] for e in file['net/params/name']])
value = np.squeeze([file[e[0]][:] for e in file['net/params/value']])
weightDecay = np.squeeze([file[e[0]][:] for e in file['net/params/weightDecay']])




[np.array(file[n[0]][:]).shape for n in file['net/layers/params']]
[(2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2,), (2, 1), (3, 1), (2,), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2,), (2, 1), (3, 1), (2,), (2, 1), (3, 1), (2,), (2,), (2,)]



file['net/params/learningRate']
# <HDF5 dataset "learningRate": shape (70, 1), type "|O">


