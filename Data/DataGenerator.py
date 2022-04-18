import pandas as pd
import os
import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(512, 512, 3),
                 shuffle=True):
        """
        df : Dataset dataframe
        X_col : a dictionary which has a mapping of key with actual column names
              : Currently the key and value of the dictionary are same as the dataframe columns
        y_col : Similar to X_columns. Contains a mapping to columns in dataframe for the prediction features
        input_size : Shape of the image
        """
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
    
    def __len__(self):
        return int(self.n / self.batch_size)
    
    def __get_input(self, path):
        """
        A helper function which returns the image array from image path
        """
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize(image_arr,(self.input_size[0], self.input_size[1])).numpy()
        return image_arr/255
    
    def __get_output(self, label, num_classes):
        """
        A helper function which converts numerical class labels to one hot encoded vectors
        """
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
            
    def __get_data(self, batches):
        """
        A helper function which returns the following a tuple of X and y
        X - > Image array of shape (Batch_size, H, W)
        y -> A tuple of class label and a list of bounding box coordinates
        """
        path_batch = batches[self.X_col['path']]
        class_batch = batches[self.y_col['class']]
        bb_batch = batches[[self.y_col['xmin'], self.y_col['ymin'], self.y_col['xmax'], self.y_col['ymax']]]

        X_batch = np.asarray([self.__get_input(x_path) for x_path in path_batch])
        return X_batch, (class_batch, bb_batch)
    
    def __getitem__(self, index):
        """
        Returns a batch of X and y data
        X - > Image array of shape (Batch_size, H, W)
        y -> A tuple of class label and a list of bounding box coordinates
        """
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y