# Class for taking in TRAINED TENSORFLOW models and outputting a clustered model
# Weights are clustered according to the paper, network is then finetuned

from gc import callbacks
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np

class Prune:
    def __init__(self, trained_model) -> None:
        """
        Class to abstract away Pruning functionality

        Takes in trained tensorflow model and performs pruning on it to reduce 
        space

        Params:
        ---------------
        trained_model: a trained tensorflow model.E.g. ResNet-20 trained on 
        CIFAR-10

        history: history of the model after fine tuning occurs

        Usage:
        ------------
        Construct an instance with a pretrained tensorflow model. Then call 
        prune()
        """

        self.model = trained_model
    
    def prune(self,train_images, train_labels,batch_size=128,fine_tune_epochs=1, final_sparsity=0.8):

        """
        Prunes the network to the specified sparsity

        Params:
        --------

        train_images: images to train on

        train_labels: one-hot label vectors

        final_sparsity: the final sparsity we want to achieve for the network, 
        i.e., what percent of the connections do we want pruned
        """


        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        num_images = train_images.shape[0]
        end_step = np.ceil(num_images / batch_size).astype(np.int32) * fine_tune_epochs

        # Define model for pruning.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                    final_sparsity=final_sparsity,
                                                                    begin_step=0,
                                                                    end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)

        model_for_pruning.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        model_for_pruning.fit(train_images,train_labels,batch_size=batch_size,epochs=fine_tune_epochs,callbacks=callbacks)

        self.model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        

    def prune_new(self, train_generator, batch_size=128,fine_tune_epochs=1, final_sparsity=0.8):

        """
        Prunes the network to the specified sparsity

        Params:
        --------

        train_images: images to train on

        train_labels: one-hot label vectors

        final_sparsity: the final sparsity we want to achieve for the network, 
        i.e., what percent of the connections do we want pruned
        """


        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude


        end_step = np.ceil(len(train_generator)).astype(np.int32) * fine_tune_epochs

        # Define model for pruning.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                    final_sparsity=final_sparsity,
                                                                    begin_step=0,
                                                                    end_step=end_step)
        }

        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)

        model_for_pruning.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
        num_classes = 3

        for step, (x_batch_train, y_batch_train) in enumerate(train_generator):
            if step >= 3:
                break
            print('Current step', step)
            true_labels, true_bboxes = y_batch_train
            true_labels = tf.keras.utils.to_categorical(true_labels, num_classes)
            print(true_labels.shape)
            model_for_pruning.fit(x_batch_train,true_labels,batch_size=batch_size,epochs=fine_tune_epochs,callbacks=callbacks)

        self.model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        

class Cluster:
    def __init__(self, trained_model) -> None:
        """
        Class to abstract away the clustering functionality

        Takes in a trained tensorflow model and performs weight sharing on it to
        reduce space.

        Params:
        -----------
        trained_model: a trained tensorflow model. E.g. ResNet-20 
        trained on CIFAR-10

        Fields:
        ------------
        model: a tensorflow model. This model will be affected by calls to 
        cluster() and finetune()

        history: history of the model after fine tuning occurs

        Usage:
        ------------
        Construct an instance with a pretrained tensorflow model. Then call 
        cluster() and finetune() 
        """

        self.model = trained_model

        """
        For internal use
        """
        self.clustering_flag = False #Set to true after clustering is performed

    def cluster(self, num_clusters=16):
        """
        Performs clustering/weight sharing on the pre-trained model

        Parameters:
        -----------
        num_clusters: the number of unique weights the new network will have
        """

        cluster_weights = tfmot.clustering.keras.cluster_weights
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
        
        # We keep centroid initialization scheme to be linear, 
        # just like in the paper

        clustering_params = {
        'number_of_clusters': num_clusters,
        'cluster_centroids_init': CentroidInitialization.LINEAR
        }

        # Cluster the model
        self.model = cluster_weights(self.model, **clustering_params)

        self.clustering_flag = True
    
    def finetune(self,x_train, y_train,batch_size=256,learning_rate=1e-5,epochs=1):
        """
        Performs fine tuning using the given training data. Will only run if you
        have clustered the model. Uses Adam optimizer

        Parameters:
        ----------------
        x_train: the training data
        y_train: the labels for the training data
        batch_size: the batch_size
        learning_rate: the learning rate, keep this low because we are just 
        finetuninrg
        epochs: the number of epochs to train for. Keep low for finetuning         
        """

        if not self.clustering_flag:
            raise RuntimeError("Need to cluster the weights before fine tuning can take place. Call Cluster.cluster()")

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

        self.history = self.model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)

        self.model = tfmot.clustering.keras.strip_clustering(self.model)


class Quantize:
    def __init__(self, trained_model) -> None:
        """
        Class to abstract away quantization aware training

        Takes in trained tensorflow model and performs quantization on it to reduce 
        space

        Params:
        ---------------
        trained_model: a trained tensorflow model.E.g. ResNet-20 trained on 
        CIFAR-10

        history: history of the model after fine tuning occurs

        Usage:
        ------------
        Construct an instance with a pretrained tensorflow model. Then call 
        quantize() followed by finetune()
        """

        self.model = trained_model

        """
        For internal use only
        """
        self.quantized_flag = False
        self.finetuned_flag = False
    
    def quantize(self):
        """
        Quantizes network weights and recompiles the model
        """
        quantize_model = tfmot.quantization.keras.quantize_model

        q_model = quantize_model(self.model)

        # `quantize_model` requires a recompile.
        q_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        self.model = q_model

        self.quantized_flag = True

    def finetune(self, x_train, y_train, batch_size=256, epochs=1):
        """
        Performs fine-tuning with newly quantized weights

        Parameters:
        -------------
        x_train: the training images
        y_train: the labels for the training images
        batch_size: the batch size
        epochs: number of epochs to fine tune for
        """
        
        if self.quantized_flag:
            self.model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs)

            self.finetuned_flag = True
        else:
            raise RuntimeError('Must call quantize() before calling finetune()')