from pickle import FALSE
from pickletools import optimize
from AlexNet import AlexNet
from ResNet import ResNet
from VGG import VGG16, VGG19
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, losses

SKIP_ALEXNET = True
SKIP_RESNET = False
SKIP_VGG = True
SKIP_MNIST = True

ALEXNET_MNIST_MSG = 'Skipping training MNIST on AlexNet to save time'

# alex = AlexNet((227,227,3), 10)

# vgg16 = VGG16((32,32,3), 10)
# vgg19 = VGG19((32,32,3),10)

class TestAlexNet(unittest.TestCase):

    def setUp(self) -> None:
        pass
    
    def tearDown(self) -> None:
        pass

    @unittest.skipIf(SKIP_ALEXNET and SKIP_MNIST, ALEXNET_MNIST_MSG)
    def test_mnist(self):
        (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
        x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
        x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        x_test = tf.expand_dims(x_test, axis=3, name=None)
        x_train = tf.repeat(x_train, 3, axis=3)
        x_test = tf.repeat(x_test, 3, axis=3)
        x_val = x_train[-2000:,:,:,:]
        y_val = y_train[-2000:]
        x_train = x_train[:-2000,:,:,:]
        y_train = y_train[:-2000]

        alex = AlexNet(x_train.shape[1:], num_classes=10)

        alex.model.compile(optimizer='adam',
         loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        # Trains model on data
        # I tested it and it achieves greater than 98% validation accuracy
        history = alex.model.fit(x_train, y_train, batch_size=64, epochs=1,
                                             validation_data=(x_val, y_val))
class TestResNet(unittest.TestCase):
    def setUp(self) -> None:
        pass
    
    def test(self):
        pass

    def test2(self):
        pass
# Run the tests
if __name__=='__main__':
    unittest.main()