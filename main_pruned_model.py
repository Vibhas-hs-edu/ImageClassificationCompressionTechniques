from Models.ResNetPruned import ResNetPruned
import tensorflow as tf
import os
from pathlib import Path

input_shape = (512, 512, 3)
best_resnet18_epoch = 10
best_resnet50_epoch = 5
num_classes = 3
depth = 20


def prune_and_save_model(uncompressed_model_file, pruned_model_name, save_path, batch_size):
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)

    r_comp= ResNetPruned(input_shape = input_shape, depth = depth, num_classes = num_classes, model_name = pruned_model_name)
    r_comp.load_and_prepare_model(model_file = uncompressed_model_file, batch_size = batch_size)
    pruned_model = r_comp.model_pruned
    pruned_model.save(save_path)

prune_and_save_model(uncompressed_model_file =  f'Results/ResNet18/ResNet_{best_resnet18_epoch}.h5', 
                        pruned_model_name = "ResNet18_Pruned",
                        save_path = f'Results/ResNet18/ResNet_Pruned_{best_resnet18_epoch}.h5',
                        batch_size = 16)

prune_and_save_model(uncompressed_model_file =  f'Results/ResNet50/ResNet50_{best_resnet50_epoch}.h5', 
                        pruned_model_name = "ResNet50_Pruned",
                        save_path = f'Results/ResNet50/ResNet_Pruned_{best_resnet50_epoch}.h5',
                        batch_size = 16)