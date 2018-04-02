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
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'



img_rows, img_cols = 28, 28

(data, labels), (test, testlabels)  = mnist.load_data()

forpre_labels = testlabels

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

#####################################$

#####################################$
model.add(Flatten(input_shape=(28,28, 1)))
#####################################$
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', # mean_squared_error
            optimizer='adagrad', # adagrad vs adadelta
            metrics=['accuracy'])

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

model.fit_generator(datagen.flow(data, labels, batch_size=200), epochs=1, validation_data=(test, testlabels))
# model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.fit(data, labels, epochs=3, batch_size=200, validation_data=(test, testlabels))

score = model.evaluate(test, testlabels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#########################################################$


##########################################################$

predicted_classes = model.predict_classes(test)

for i in range(0, 10):
	print(predicted_classes[i])
	print(forpre_labels[i])

# see which we predicted correctly and which not
correct_indices = []
incorrect_indices = []

for i in range(0, 10000):
	if predicted_classes[i] == forpre_labels[i]:
		correct_indices .append(i)
	else:
		incorrect_indices.append(i)
		
		
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
print(correct_indices[:9])
print(incorrect_indices[:9])
# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (14,28)

plt.figure(1)

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], forpre_labels[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], forpre_labels[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.savefig('fig1.png')
