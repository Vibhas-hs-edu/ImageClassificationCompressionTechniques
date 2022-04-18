"""
Abstracts away conversion methods for:
1. Converting from tensorflow to tensorflow.js
"""
import tensorflowjs as tfjs

class Convert:
    def __init__(self, trained_model) -> None:
        self.model = trained_model

    def to_tfjs(self, save_dir):
        """
        Converts tensorflow model to Tensorflow.js model in specified directory

        Paramters:
        ----------

        save_dir: the directory to save the model to
        """

        tfjs.converters.save_keras_model(self.model, save_dir)