import keras
from keras.models import Sequential
from keras.datasets import mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data();

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
