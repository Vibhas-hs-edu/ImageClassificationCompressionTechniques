import tensorflow as tf
import keras
from torch import classes 

class VGG16:
    def __init__(self, input_shape, num_classes) -> None:
        self.model = tf.keras.applications.vgg16.VGG16(weights=None, input_shape = input_shape, classes=num_classes)

class VGG19:
    def __init__(self, input_shape, num_classes) -> None:
        self.model = tf.keras.applications.vgg19.VGG19(weights=None, input_shape = input_shape, classes=num_classes)