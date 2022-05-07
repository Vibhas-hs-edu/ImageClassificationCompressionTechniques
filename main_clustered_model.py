from Models.Compression import Cluster
from Models.ResNetClustered import ResNetCluster
import tensorflow as tf
import os
from pathlib import Path

input_shape = (512, 512, 3)
best_resnet18_epoch = 9
best_resnet50_epoch = 4
num_classes = 3
depth = 20
num_clusters = 16

def cluster_and_save_model(uncompressed_model_file, cluster_model_name, save_path, num_clusters, batch_size):
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)

    r_comp= ResNetCluster(input_shape = input_shape, depth = depth, num_classes = num_classes, model_name = cluster_model_name)
    r_comp.load_and_prepare_model(model_file = uncompressed_model_file, batch_size = batch_size, num_clusters = num_clusters)
    clustered_model = r_comp.model_cluster
    clustered_model.save(save_path)

num_clusters = [32, 64, 128]

for num_cluster in num_clusters:
    print(f'Generating clustering for {num_cluster} for ResNet18')
    cluster_and_save_model(uncompressed_model_file =  f'Results/ResNet18/ResNet_{best_resnet18_epoch}.h5', 
                        cluster_model_name = "ResNet18_Cluster",
                        save_path = f'Results/ResNet18/ResNet_Cluster_{num_cluster}_{best_resnet18_epoch}.h5',
                        batch_size = 16, num_clusters = num_cluster)
    print(f'Finished clustering for ResNet18')
    print()
    print(f'Generating clustering for {num_cluster} for ResNet50')
    cluster_and_save_model(uncompressed_model_file =  f'Results/ResNet50/ResNet50_{best_resnet50_epoch}.h5', 
                        cluster_model_name = "ResNet50_Cluster",
                        save_path = f'Results/ResNet50/ResNet50_Cluster_{num_cluster}_{best_resnet50_epoch}.h5',
                        batch_size = 8, num_clusters = num_cluster)
    print(f'Finished clustering for {num_cluster} for ResNet50')
    print()