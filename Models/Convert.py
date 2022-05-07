"""
Abstracts away conversion methods for:
1. Converting from tensorflow to tensorflow.js
2. Converting from tensorflow to TFLite
"""
import tensorflowjs as tfjs
import tensorflow as tf


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


    def to_tflite(self, filename):
        """
        Converts the current model to a tflite model, performs post-training 
        quantization, and writes the model to the specified file

        Parameters:
        ---------------
        filename: the name of the file you want the model to be written to. Can
        include path information, just don't need to include the .tflite 
        extension
        """

        if self.finetuned_flag and self.quantized_flag:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            quantized_tflite_model = converter.convert()



            filename_with_extension = filename+'.tflite'

            with open(filename_with_extension, 'wb') as f:
                f.write(quantized_tflite_model)
                
        else:
            raise RuntimeWarning('Must call quantize() and then finetune() before creating tflite file')