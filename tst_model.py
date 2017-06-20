from __future__ import division, print_function, absolute_import
from tflearn.data_utils import build_hdf5_image_dataset
import h5py
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle
import tensorflow as tf
import operator
from os import listdir
from os.path import isfile, join,isdir

from os import walk

f ={}
for (dirpath, dirnames, filenames) in walk('256_ObjectCategories'):
    for d in dirnames:
        tmp = str.split(d,'.')
        f.__setitem__(int(tmp[0]),tmp[1])
    # f.extend(dirnames)
    break

# onlyfiles = [f for f in listdir('256_ObjectCategories') if isdir(join('256_ObjectCategories', f))]
# print(onlyfiles)

dataset_file = 'test'
build_hdf5_image_dataset(dataset_file, image_shape=(388, 374), mode='folder', output_path='test_dataset.h5', categorical_labels=True)

h5f = h5py.File('test_dataset.h5', 'r')
X = h5f['X']
Y = h5f['Y']


# Build network
network = input_data(shape=[None, 388, 374, 3], dtype=tf.float32)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 20, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network)
model.load('cifarcnnv8.tflearn')
res = model.predict(X)
print(res)
# max_index, max_value = max(enumerate(res[0]), key=operator.itemgetter(1))
# filekey = max_index + 1
# sortd = sorted(res[0])
# print(max_index,max_value)
# print(f.get(filekey))
