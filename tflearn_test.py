from __future__ import division, print_function, absolute_import

import os

from tflearn.data_utils import build_hdf5_image_dataset
import h5py
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization, l2_normalize
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle

# Prepare train data (Run in first run only)
dataset_file = '256_ObjectCategories'
filename = 'train_dataset_64.h5'
img_dim = 64
if not os.path.exists(filename):
    build_hdf5_image_dataset(dataset_file, image_shape=(img_dim, img_dim), mode='folder', output_path=filename,
                             categorical_labels=True)

# Open Train data
h5f = h5py.File('train_dataset_%s.h5' % str(img_dim), 'r')
X = h5f['X']
Y = h5f['Y']

print(X.__len__())
print(Y.__len__())
print(X.shape)
print(Y.shape)

# 257. class clutter

X, Y = shuffle(X, Y)
print(len(X), len(Y))

# trX = X[0:21424,:]
# trY = Y[0:21424,:]
# valX = X[21424:24484,:]
# valY = Y[21424:24484,:]
# tsX = X[24484:30608,:]
# tsY = Y[24484:30608,:]

trX = X[0:24484:]
trY = Y[0:24484, :]
tsX = X[24484:30608, :]
tsY = Y[24484:30608, :]

print(len(trX), len(tsX))

trX = trX.reshape([-1, img_dim, img_dim, 3])
tsX = tsX.reshape([-1, img_dim, img_dim, 3])

trY = trY.reshape([-1, 257])
tsY = tsY.reshape([-1, 257])

# # Building convolutional network
# network = input_data(shape=[None, 128, 128, 3], name='input')
# network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 3,)
# network = local_response_normalization(network)
# network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 3)
# network = local_response_normalization(network)
# network = fully_connected(network, 128, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 256, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 257, activation='softmax')
# network = regression(network, optimizer='adam', learning_rate=0.01,
#                      loss='categorical_crossentropy', name='target')
# # Training
# model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit({'input': trX}, {'target': trY}, n_epoch=5,batch_size = 128,
#            validation_set=({'input': valX}, {'target': valY}),
#            snapshot_step=100, show_metric=True, run_id='convnet')
#
# model.save('convnet.tflearn')


# # Build network
# network = input_data(shape=[None, 128, 128, 3], dtype=tf.float32)
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 64, 3, activation='relu')
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = fully_connected(network, 512, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 257, activation='softmax')
# network = regression(network, optimizer='adam',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001)

# # Build network
# network = input_data(shape=[None, 64, 64, 3], dtype=tf.float32)
# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = fully_connected(network, 265, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 257, activation='softmax')
# network = regression(network, optimizer='adam',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001)


network = input_data(shape=[None, img_dim, img_dim, 3])
network = conv_2d(network, 96, 7, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = batch_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = batch_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = batch_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = batch_normalization(network)
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = batch_normalization(network)
network = fully_connected(network, 4096, activation='relu')
network = fully_connected(network, 4096, activation='relu')

network = fully_connected(network, 257, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy')

# network = input_data(shape=[None, 128, 128, 3], dtype=tf.float32)
# network = conv_2d(network, 128, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 128, 3, activation='relu')
# network = conv_2d(network, 128, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = fully_connected(network, 512, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 257, activation='softmax')
# network = regression(network, optimizer='adam',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trX, trY, n_epoch=100, shuffle=True, validation_set=0.1,
          show_metric=True, batch_size=128, run_id='object_365_cnn_v10')
model.save('object_365_cnn_v10.tflearn')

score = model.evaluate(tsX, tsY)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))

h5f.close()
