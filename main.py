from train.train import train_model
from Data.prepare_data import get_generator
from Models.AlexNet import AlexNet
from Models.ResNet import ResNet
import tensorflow as tf
import logging

input_shape = (512, 512, 3)
num_classes = 3
depth = 20

resnet_model = ResNet(input_shape = input_shape, depth = 20, num_classes = num_classes)
alexnet_model = AlexNet(input_shape = input_shape, num_classes = num_classes)

train_generator = get_generator("Train", input_shape)
val_generator = get_generator("Val", input_shape)
test_generator = get_generator("Test", input_shape)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

train_model(train_dataset = train_generator, val_dataset = val_generator, model = alexnet_model, loss_fn = loss_fn, filename = "AlexNetLogs.txt")