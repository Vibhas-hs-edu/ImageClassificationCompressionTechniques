from tensorflow import keras
import tensorflow as tf
import keras
from Data.prepare_data import get_generator
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from Models.base_model import BaseModel
from Models.Compression import Cluster, Prune

class ResNetCluster(BaseModel):
    def __init__(self, input_shape, depth, num_classes=10, save_dir = 'ResNet', model_name = 'ResNet') -> None:
        super().__init__(model_name, save_dir)
        self.input_shape = input_shape
        self.model_cluster = None
    
    def load_and_prepare_model(self, model_file, batch_size, num_clusters):
        assert os.path.exists(model_file), "Model file doesn't exist"
        uncompressed_model = tf.keras.models.load_model(model_file)
        c = Cluster(uncompressed_model)
   
        train_generator = get_generator("Train", self.input_shape, batch_size = batch_size)
        c.cluster(num_clusters = num_clusters)
        c.finetune_generator(train_generator = train_generator, batch_size = batch_size)
        self.model_cluster = c.model