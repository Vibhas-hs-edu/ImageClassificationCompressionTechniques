from statistics import mode
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import logging

DEF_EPOCHS = 100
DEF_MODEL = None
DEF_OPTIMIZER = keras.optimizers.SGD(learning_rate=1e-3)

DEF_TRAIN_METRIC = keras.metrics.SparseCategoricalAccuracy()
DEF_VAL_METRIC = keras.metrics.SparseCategoricalAccuracy()

def train_model(train_dataset, val_dataset, model, loss_fn, epochs = DEF_EPOCHS, optimizer = DEF_OPTIMIZER, train_acc_metric = DEF_TRAIN_METRIC
                ,val_acc_metric = DEF_VAL_METRIC, filename = 'example.log', save_dir = 'Models'):

    logging.basicConfig(filename = filename, level=logging.INFO)
    for epoch in range(epochs):
        start_time = time.time()
        logging.info(f"Start of epoch {epoch}")
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            true_labels, true_bboxes = y_batch_train
            with tf.GradientTape() as tape:
                preds = model.model(x_batch_train, training=True)
                loss_value = loss_fn(true_labels, preds)
            grads = tape.gradient(loss_value, model.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(true_labels, preds)

            # Log every 200 batches.
            if step % 20 == 0:
                logging.info(f"Training loss for one batch at step {step}: {float(loss_value)}")
                logging.info(f"Seen so far: %d samples f{((step + 1) * len(x_batch_train))}")

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        logging.info(f"Training acc over epoch: {float(train_acc)}")

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()
        if epoch % 20 == 0:
            model.save_model(model.model, epoch)

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_preds = model(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_preds)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        logging.info(f"Validation acc: {float(val_acc)}")
        logging.info(f"Time taken: {time.time() - start_time}")