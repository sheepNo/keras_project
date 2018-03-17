#!/usr/bin/env python3
"""
Our implementation
"""

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.datasets import mnist

import numpy as np
# import chainer


# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# model = Sequential()
# model.add(Dense(unit=64, activation='relu', input_dim=768))
# model.add(Dense(units=10, activation='softmax'))
#
# # train[i] represents i-th data, there are 60000 training data
# # test data structure is same, but total 10000 test data
# # print('len(train), type ', len(train), type(train))
# # print('len(test), type ', len(test), type(test))
# # print('train[0]', train[0])
# #
# # print('\n\n\n')
# #
# # print('train[0][0]', train[0][0].shape)
# # np.set_printoptions(threshold=np.inf)
# #
# # print('\n\n\n')
# #
# # print(train[0][1])
# #
#
#
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
img_rows, img_cols = 28, 28

(data, labels), (test, testlabels)  = mnist.load_data()

# cheating
# if K.image_data_format() == 'channels_first':
data = data.reshape(data.shape[0], 784)
test = test.reshape(test.shape[0], 784)
input_shape = (1, img_rows, img_cols)
# else:
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # input_shape = (img_rows, img_cols, 1)


# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='sparse_categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5, batch_size=32)
model.fit(data, labels, epochs=30, batch_size=1000)  # starts training
