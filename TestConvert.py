from Models.Convert import Convert
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, losses
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers


RES20_FILE = 'res20.h5'


class ConvertToJs(unittest.TestCase):

    def test1(self):
        
        model = tf.keras.models.load_model(RES20_FILE)

        convert = Convert(model)

        convert.to_tfjs('Models')



# Run the tests
if __name__=='__main__':
    unittest.main()