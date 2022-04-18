import pandas as pd
import Data.data_constants as data_constants
import os
import tensorflow as tf
import numpy as np
from Data.DataGenerator import DataGenerator


data_folder = data_constants.DATA_FOLDER
images_folder = os.path.join(data_folder, '2A_images')
train_file = os.path.join(data_folder, 'train_COVIDx_CT-2A.txt')
test_file = os.path.join(data_folder, 'test_COVIDx_CT-2A.txt')
val_file = os.path.join(data_folder, 'val_COVIDx_CT-2A.txt')

assert os.path.exists(data_folder), 'Data folder must be downloaded first'
assert os.path.exists(images_folder), 'Images was not downloaded properly'

assert os.path.exists(train_file)
assert os.path.exists(test_file)
assert os.path.exists(val_file)

col_names = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
train_ds = pd.read_csv(train_file, sep = ' ', names = col_names)
test_ds = pd.read_csv(test_file, sep = ' ', names = col_names)
val_ds = pd.read_csv(val_file, sep = ' ', names = col_names)

train_ds['filename'] =  train_ds['filename'].apply(lambda x: os.path.join(images_folder, x))
val_ds['filename'] =  val_ds['filename'].apply(lambda x: os.path.join(images_folder, x))
test_ds['filename'] =  test_ds['filename'].apply(lambda x: os.path.join(images_folder, x))

def get_generator(generator_type, input_size, batch_size = 32):
    batch_size = 64
    x_map = {'path':'filename'}
    y_map = {'class': 'class', 'xmin': 'xmin', 'ymin' : 'ymin', 'xmax' : 'xmax', 'ymax' : 'ymax'}
    if generator_type == "Train":
        train_gen = DataGenerator(train_ds,
                                X_col = x_map,
                                y_col=y_map,
                                batch_size=batch_size,
                                input_size = input_size)
        return train_gen
    elif generator_type == "Val":
        val_gen = DataGenerator(val_ds,
                                X_col = x_map,
                                y_col = y_map,
                                batch_size=batch_size,
                                input_size = input_size)
        return val_gen
    elif generator_type == "Test":
        test_gen = DataGenerator(test_ds,
                                X_col = x_map,
                                y_col = y_map,
                                batch_size=batch_size,
                                input_size = input_size)
        return test_gen
    else:
        raise ValueError("Generator type must be one of Train, Validation or Test")