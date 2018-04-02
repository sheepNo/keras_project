#!/usr/bin/env python3
"""
Our implementation
"""

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, GaussianDropout
from keras.optimizers import *
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
# import chainer


# Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# model = Sequential()
# model.add(Dense(units=64, activation='relu', input_dim=768))
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

datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1)

# cheating
# if K.image_data_format() == 'channels_first':
data = data.reshape(data.shape[0], img_rows, img_cols, 1)
test = test.reshape(test.shape[0], img_rows, img_cols, 1)
input_shape = (1, img_rows, img_cols)
# else:
    # x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # input_shape = (img_rows, img_cols, 1)

labels, testlabels = to_categorical(labels, 10), to_categorical(testlabels, 10)

model = Sequential()


# model.add(Conv2D(40, kernel_size=(2,2), activation='relu', input_shape=(28,28, 1)))
# model.add(Dropout(0.25))
# model.add(Conv2D(20, kernel_size=(4,4), activation='sigmoid' ))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(5, kernel_size=(3,3), strides=(2,1), activation='relu', input_shape=(28,28, 1)))
model.add(GaussianDropout(0.05))

model.add(MaxPooling2D(pool_size=(2,1))) # 2,1 is pretty good

# add dropouts
model.add(Conv2D(10, kernel_size=(6,6), strides=(3,3), activation='sigmoid'))
model.add(GaussianDropout(0.05))
# model.add(Conv2D(128, kernel_size=(6,6), activation='relu'))

# model.add(Conv2D(28, kernel_size=(6,6), strides=(3,3), activation='sigmoid'))
# model.add(AveragePooling2D(pool_size=(2,2))) # 2,1 is pretty good
model.add(Flatten())
# model.add(Dropout(0.20))
# add dropouts
# model.add(Conv2D(16, kernel_size=(8,8), activation='relu'))

model.add(Dense(10, activation='softmax'))

# # This returns a tensor
# inputs = Input(shape=(784,))
#
# # a layer instance is callable on a tensor, and returns a tensor
# x = Dense(256, activation='relu')(inputs)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='categorical_crossentropy', # mean_squared_error
            optimizer='adagrad', # nadam
            metrics=['accuracy'])
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=5, batch_size=32)
model.fit_generator(datagen.flow(data, labels, batch_size=200), epochs=20, validation_data=(test, testlabels))
# model.fit(data, labels, epochs=3, batch_size=200, validation_data=(test, testlabels))

score = model.evaluate(test, testlabels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
