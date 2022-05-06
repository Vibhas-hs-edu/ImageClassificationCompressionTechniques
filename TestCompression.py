from Models.Convert import Convert
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, losses
from Models.Compression import Cluster, Prune, Quantize

SKIP_MSG = 'Already ran'

SKIP_PRUNE_TESTS = True
SKIP_CLUSTER_TESTS=True

RES20_FILE = 'res20.h5'


class CompressionTests(unittest.TestCase):

    def setUp(self) -> None:
        self.model = tf.keras.models.load_model(RES20_FILE)

        num_classes = 10

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        # Input image dimensions.
        input_shape = x_train.shape[1:]

        # Normalize data.
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255

        self.y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        self.y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    @unittest.skipIf(SKIP_PRUNE_TESTS,SKIP_MSG)
    def testPrune(self):

        _, baseline_acc = self.model.evaluate(self.x_test,self.y_test)
        baseline_acc *= 100

        print(f"Baseline pruning accuracy is: {baseline_acc: 0.2f}")

        # PRUNE
        p = Prune(self.model)
        p.prune(self.x_train,self.y_train)

        p.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        _, new_acc = p.model.evaluate(self.x_test,self.y_test)
        new_acc *= 100

        print(f"Accuracy after pruning is {new_acc : 0.2f}")

        self.assertLess(baseline_acc-new_acc,2, f"The accuracy dropped by {baseline_acc-new_acc : 0.2f} which is more than 2")

        # Save the pruned model
        p.model.save('resnet20/prunedV2')

    @unittest.skipIf(SKIP_CLUSTER_TESTS,SKIP_MSG)
    def testCluster(self):
        results = self.model.evaluate(self.x_test,self.y_test)

        prev_acc = results[1] *100
        print(f"Accuracy before clustering: {prev_acc : 0.2f}%")

        c = Cluster(self.model)
        c.cluster(16)
        c.finetune(self.x_train,self.y_train)

        # Have to recompile before re-evaluating
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        c.model.compile(loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

        new_results = c.model.evaluate(self.x_test,self.y_test)
        new_acc = new_results[1] *100 

        print(f"Accuracy after clustering: { new_acc: 0.2f}%")

        decrease_in_acc = prev_acc-new_acc

        self.assertLess(decrease_in_acc,2,"Accuracy decreased by greater than 2% after cluseting")

        c.model.save('resnet20/clustered/clusteredv2.h5')

    def testPruneAndCluster(self):
        results = self.model.evaluate(self.x_test,self.y_test)

        prev_acc = results[1] *100
        print(f"Accuracy before pruning and clustering: {prev_acc : 0.2f}%")

        p = Prune(self.model)
        p.prune(self.x_train,self.y_train,fine_tune_epochs=2)

        p.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        _, new_acc = p.model.evaluate(self.x_test,self.y_test)
        new_acc *= 100

        print(f"Accuracy after pruning is {new_acc : 0.2f}")
        

        c = Cluster(p.model)
        c.cluster(32)
        c.finetune(self.x_train,self.y_train,epochs=4)

        # Have to recompile before re-evaluating
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        c.model.compile(loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

        new_results = c.model.evaluate(self.x_test,self.y_test)
        new_acc = new_results[1] *100 

        print(f"Accuracy after pruning and clustering: { new_acc: 0.2f}%")

        decrease_in_acc = prev_acc-new_acc

        c.model.save('resnet20/prunedAndClustered.h5')

        self.assertLess(decrease_in_acc,2,"Accuracy decreased by greater than 2% after pruning and clustering")

       
# Run the tests
if __name__=='__main__':
    unittest.main()