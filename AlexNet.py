#--------------- CREDIT--------------------------------------#
# Code for building the model is slightly modified from the following link 
# https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8

import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models, losses

class AlexNet:
    def __init__(self, input_shape, num_classes) -> None:
        """
        Class to abstract away implementation of AlexNet

        Parameters:
        ------------- 
        input_shape (3-tuple): Shape of the input. Must have 3 channels and be 
        at least 32x32. For example, 32x32x3 is valid, 240x240x3 is valid,
        5x5x3 is not valid. NOTE: all inputs are reshaped to 224x224x3 to be 
        compatible with AlexNet architecture 

        num_classes (int): number of output classes there are in the data

        Attributes:
        --------------
        model: the AlexNet model

        """
        model = keras.models.Sequential()
        model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=input_shape))
        model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
        model.add(layers.Lambda(tf.nn.local_response_normalization))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(3, strides=2))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
        model.add(layers.Activation('relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))

        self.model = model
        self.model.compile()