# CREDIT: https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8

import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models, losses

class AlexNet:
    def __init__(self, input_shape, num_classes) -> None:
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