import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.datasets import mnist
from keras import optimizers
import numpy as np
from keras import backend as K

batch_size = 500
num_classes = 10
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28
#input is accepted
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    print("channels_first!!")
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    print("channels_last!!")

#this is needed for validation_data to work, which reshapes the image into 4 dimensions
#in the original program it checked for the CHANNELS being at first or at last position in image
#here its channels_last by default

#x_train = x_train.reshape(x_train.shape[0], 784)
#x_test = x_test.reshape(x_test.shape[0], 784)

#not needed to reshape anymore

input_shape = ( img_rows, img_cols, 1)

#inputs = Input(shape=(784,))
#a tensor is returned above
#up to this is exactly the same as NN_MNIST

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to 10 class matrices,as there are 10 labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#^^^ split the labels into its classes, here it's 10

model = Sequential()
model.add(Dense(28, input_shape=input_shape, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Flatten())
#the validation_data didn't work without using the Flatten , why??
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
#Test loss: 0.04506711347810924
#Test accuracy: 0.985790002155304


#available optimizers= sgd,
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#check out the link in messenger

# starts training
print("CHECK!")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#validation data works on the TEST DATA only, not used during training, expects 4 dimensions
#by default, BUT WHY 4????
#evaluate test performed from the training above
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
