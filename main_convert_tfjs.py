import sys
from Models.Convert import Convert
import tensorflow as tf


def convert_driver(input_filepath, output_dir):
    """
    Converts the .h5 model at the input_filepath location to a tfjs model
    Stores the tfjs model in directory output_dir

    Params
    ----------

    input_filepath: the path to the .h5 file to convert

    output_dir: the path to the output directory we want to store the json model in
    """

    model = tf.keras.models.load_model(input_filepath)
    convert = Convert(model)
    print('Converting now')
    convert.to_tfjs(output_dir)

if __name__=='__main__':

    if len(sys.argv) != 3:
        raise ValueError('Need an input file path and an output directory')

    input_filepath = sys.argv[1]
    output_dir = sys.argv[2]

    convert_driver(input_filepath, output_dir)
