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
import matplotlib.pyplot as plt

import numpy as np
from keras.utils import plot_model




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

#####################################$
model.add(Conv2D(32, kernel_size=(2,3), activation='relu', input_shape=(28,28, 1)))
model.add(BatchNormalization())
#model.add(Dropout(0.10))
model.add(Dropout(0.50))
#####################################$
model.add(Conv2D(32, kernel_size=(4,6), activation='relu', input_shape=(28,28, 1)))
model.add(BatchNormalization())
#model.add(Dropout(0.10))
model.add(Dropout(0.50))
#####################################$
model.add(AveragePooling2D(pool_size=(2,2))) # 2,1 is pretty good
#####################################$
model.add(Conv2D(64, kernel_size=(6,9), activation='sigmoid'))
model.add(BatchNormalization())
#model.add(Dropout(0.10))
model.add(Dropout(0.50))
#####################################$
model.add(MaxPooling2D(pool_size=(2,2))) # 2,1 is pretty good
#####################################$
model.add(Flatten())
#####################################$
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', # mean_squared_error
            optimizer='adagrad', # adagrad vs adadelta
            metrics=['accuracy'])

# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

history=model.fit_generator(datagen.flow(data, labels, batch_size=200), epochs=2, validation_data=(test, testlabels))
# model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.fit(data, labels, epochs=3, batch_size=200, validation_data=(test, testlabels))

score = model.evaluate(test, testlabels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
##############plot_model(model, to_file='model.png')############################################Copied from net###########
plot_model(model, to_file='model.png')
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
