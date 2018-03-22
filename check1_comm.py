import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation

#(train_image, train_label), (test_image, test_label) = mnist.load_data();
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
