import tensorflow as tf
from Data.prepare_data import get_generator
from tensorflow import keras
import time

model_path_1 = "Results/ResNet18/ResNet_9.h5"
model_path_2 = "Results/ResNet50/ResNet50_4.h5"

model_path_3 = "Results/ResNet18/ResNet_Pruned_9.h5"
model_path_4 = "Results/ResNet50/ResNet_Pruned_4.h5"

model_path_5 = "Results/ResNet18/ResNet_Cluster_32_9.h5"
model_path_6 = "Results/ResNet18/ResNet_Cluster_64_9.h5"
model_path_7 = "Results/ResNet18/ResNet_Cluster_128_9.h5"


model_path_8 = "Results/ResNet50/ResNet50_Cluster_32_4.h5"
model_path_9 = "Results/ResNet50/ResNet50_Cluster_64_4.h5"
model_path_10 = "Results/ResNet50/ResNet50_Cluster_128_4.h5"

model = tf.keras.models.load_model(model_path_1)
input_shape = (512, 512, 3)
num_classes = 3
test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
test_generator = get_generator("Test", input_shape, batch_size = 16)
start_time = time.time()
for x_batch_test, y_batch_test in test_generator:
    y_batch_test_labels, y_batch_test_boxes = y_batch_test
    test_preds = model(x_batch_test, training=False)
    test_acc_metric.update_state(y_batch_test_labels, test_preds)
test_acc = test_acc_metric.result()
test_acc_metric.reset_states()
print(f"Test acc: {float(test_acc)}")
print(f"Time taken for running the model on test dataset: {time.time() - start_time}")