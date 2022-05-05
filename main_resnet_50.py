from train.train import train_model
from Data.prepare_data import get_generator
from Models.AlexNet import AlexNet
from Models.ResNet import ResNet
import tensorflow as tf
import logging

input_shape = (512, 512, 3)
num_classes = 3
depth = 50
save_dir = 'ResNet50'

resnet_model = ResNet(input_shape = input_shape, depth = depth, num_classes = num_classes, save_dir = save_dir, model_name = "ResNet50")

train_generator = get_generator("Train", input_shape)
val_generator = get_generator("Val", input_shape)
test_generator = get_generator("Test", input_shape)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

train_model(train_dataset = train_generator, val_dataset = val_generator, model = resnet_model, loss_fn = loss_fn, filename = f"{save_dir}/ResNetLogs.txt")