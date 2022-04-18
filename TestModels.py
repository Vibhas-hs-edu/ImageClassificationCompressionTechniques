import os
import tempfile
from Models.AlexNet import AlexNet
from Models.ResNet import ResNet
from Models.VGG import VGG16, VGG19
from Models.Compression import Cluster, Prune, Quantize
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, losses
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers


#---Constants for AlexNet Tests----#
SKIP_ALEXNET_TESTS = True
ALEXNET_MNIST_MSG = 'Skipping training MNIST on AlexNet to save time'
SKIP_MNIST = True
#----------------------------------#


#---Constants for ResNet Tests----#
SKIP_RESNET_TESTS = True
SKIP_RESNET_RUN = True
SKIP_SAVE_RES20 = True
RES20_NOT_SAVED = False
RES20_FILE = 'res20.h5'
#----------------------------------#

# ---Constants for Compression Tests-----#
SKIP_CLUSTER_TESTS = True
SKIP_PRUNE_TESTS = True
SKIP_QUNATIZE_TESTS = False

#-----------------------------------#

# alex = AlexNet((227,227,3), 10)

# vgg16 = VGG16((32,32,3), 10)
# vgg19 = VGG19((32,32,3),10)

@unittest.skipIf(SKIP_ALEXNET_TESTS, ALEXNET_MNIST_MSG)
class TestAlexNet(unittest.TestCase):

    def setUp(self) -> None:
        pass
    
    def tearDown(self) -> None:
        pass

    @unittest.skipIf(SKIP_ALEXNET_TESTS and SKIP_MNIST, ALEXNET_MNIST_MSG)
    def test_mnist(self):
        (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
        x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
        x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
        x_train = tf.expand_dims(x_train, axis=3, name=None)
        x_test = tf.expand_dims(x_test, axis=3, name=None)
        x_train = tf.repeat(x_train, 3, axis=3)
        x_test = tf.repeat(x_test, 3, axis=3)
        x_val = x_train[-2000:,:,:,:]
        y_val = y_train[-2000:]
        x_train = x_train[:-2000,:,:,:]
        y_train = y_train[:-2000]

        alex = AlexNet(x_train.shape[1:], num_classes=10)

        alex.model.compile(optimizer='adam',
         loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        # Trains model on data
        # I tested it and it achieves greater than 98% validation accuracy
        history = alex.model.fit(x_train, y_train, batch_size=64, epochs=1,
                                             validation_data=(x_val, y_val))
@unittest.skipIf(SKIP_RESNET_TESTS, 'Have already run resnet tests')
class TestResNet(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        print(f"The temp folder is: {self.tmpdir}")
    
    def test(self):
        #----------Create Resnet-44-------------#
        res = ResNet((32,32,3), 44)
        res.model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])

    @unittest.skipIf(SKIP_RESNET_RUN,'Already ran the model')
    def testRes101(self):
        num_classes = 10

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        # Input image dimensions.
        input_shape = x_train.shape[1:]

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        print("Should be (32,32,3)")
        print(x_train.shape[1:])

        res101 = ResNet((32,32,3), 101)
        res101.model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])

        # In one epoch, takes 30 minutes to train on my machine and gets 30% accuracy
        history = res101.model.fit(x_train, y_train,batch_size=128,epochs=1)

        # res101.model.save('/saved_models/res101.h5')
    
    @unittest.skipIf(SKIP_SAVE_RES20, 'Already saved ResNet20')
    def test_save_res20(self):

        num_classes = 10

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

        # Input image dimensions.
        input_shape = x_train.shape[1:]

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        print("Should be (32,32,3)")
        print(x_train.shape[1:])

        res20 = ResNet(input_shape=input_shape,depth=20)
        res20.model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])

        # In one epoch, takes 30 minutes to train on my machine and gets 30% accuracy
        print("Training ResNet-20")
        history = res20.model.fit(x_train, y_train,batch_size=128,epochs=1)
        
        # Saving model
        print("Saving model")


        
        save_path = os.path.join(os.getcwd(),'resnet20/1/')
        tf.saved_model.save(res20.model,save_path)
    
    @unittest.skipIf(RES20_NOT_SAVED, "Haven't saved the model yet")
    def test_load_res20(self):
        
        res20 = tf.keras.models.load_model(RES20_FILE)

        trainable = np.sum([np.prod(v.get_shape()) for v in res20.trainable_weights])
        non_trainable = np.sum([np.prod(v.get_shape()) for v in res20.non_trainable_weights])
        print("For saved resnet20:")
        print(f"Trainable: {trainable:,}\nNon-Trainable:{non_trainable:,}\nTotal:{trainable+non_trainable:,}")


        # h5 is not the proper format to save in, wants a .pb file. 
        # TODO: FIX
        save_path = os.path.join(os.getcwd(),'resnet20/1/')
        converter = tf.lite.TFLiteConverter.from_saved_model(save_path)

        # Add in weight quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        res20lite = converter.convert()

        with open('resnet20_quantized.tflite', 'wb') as f:
            f.write(res20lite)

class TestCompression(unittest.TestCase):

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

    @unittest.skipIf(SKIP_PRUNE_TESTS, 'Already ran prune tests, want to save time')
    def test_prune(self):
        """
        Test the prune class
        """
        # print('BEFORE PRUNING:')
        # self.model.summary()

        # Try pretrained model with higher accuracy:
        # print(f"Trying Pretrained ResNet50")
        # r50 = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape = self.x_train.shape[1:])
        # # r50.trainable = False

        # high_acc_model = tf.keras.models.Sequential()
        # high_acc_model.add(r50)
        # high_acc_model.add(layers.Flatten())
        # high_acc_model.add(layers.Dense(10, activation='softmax'))



        # high_acc_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        # _, r50_acc = high_acc_model.evaluate(self.x_test,self.y_test)
        # print(f"Accuracy of pretrained res50 : {100 * r50_acc : 0.2f}")

        _, baseline_acc = self.model.evaluate(self.x_test,self.y_test)
        baseline_acc *= 100

        print(f"Baseline pruning accuracy is: {baseline_acc: 0.2f}")

        # PRUNE
        p = Prune(self.model)
        p.prune(self.x_train,self.y_train)

        # print('AFTER PRUNING:')
        # p.model.summary()

        _, new_acc = p.model.evaluate(self.x_test,self.y_test)
        new_acc *= 100

        print(f"Accuracy after pruning is {new_acc : 0.2f}")

        self.assertLess(baseline_acc-new_acc,2, f"The accuracy dropped by {baseline_acc-new_acc : 0.2f} which is more than 2")





    @unittest.skipIf(SKIP_CLUSTER_TESTS, 'Already ran cluster tests, want to save time')
    def test_cluster_acc16(self):
        """
        Test the cluster class for num_clusters = 16. Test will pass if accuracy
        of the clustered model is not more than 2 percentage points less than 
        the original accuracy

        FINDINGS:
        Tested using ResNet-20 trained for one epoch. Initial accuracy was low
        and the accuracy after clustering actually increased. I suspect this 
        only happened because the accuracy was so low to begin with.

        Also, takes significantly less time to train for one epoch. 
        About 3.5 minutes on my laptop, I think it was like 30 minutes 
        originally though I could be remembering the time taken for ResNet101

        TODO: Need to have a higher accuracy model and see how much clustering 
        affects the accuracy
        """
        # First lets see what the accuracy looks like:
        
        print('BEFORE CLUSTERING:')
        self.model.summary()

        results = self.model.evaluate(self.x_test,self.y_test)

        # print(f"Results: {results}")
        prev_acc = results[1] *100
        print(f"Accuracy before clustering: {prev_acc : 0.2f}%")

        c = Cluster(self.model)
        c.cluster(16)
        c.finetune(self.x_train,self.y_train)

        print('AFTER CLUSTERING:')
        c.model.summary()

        new_results = c.model.evaluate(self.x_test,self.y_test)
        new_acc = new_results[1] *100 

        print(f"Accuracy after clustering: { new_acc: 0.2f}%")

        decrease_in_acc = prev_acc-new_acc

        self.assertLess(decrease_in_acc,2,"Accuracy decreased by greater than 2% after cluseting")

    @unittest.skipIf(SKIP_QUNATIZE_TESTS, 'Already ran quantize tests, want to save time')
    def test_quantize(self):
        """
        Tests quantization aware training
        """

        results = self.model.evaluate(self.x_test,self.y_test)

        # print(f"Results: {results}")
        prev_acc = results[1] *100
        print(f"Accuracy before quantizing: {prev_acc : 0.2f}%")


        q = Quantize(self.model)
        q.quantize()
        q.finetune(self.x_train, self.y_train)

        new_results = q.model.evaluate(self.x_test,self.y_test)
        new_acc = new_results[1] *100 

        print(f"Accuracy after quantizing: { new_acc: 0.2f}%")

        decrease_in_acc = prev_acc-new_acc

        self.assertLess(decrease_in_acc,3,"Accuracy decreased by greater than 3% after cluseting")






# Run the tests
if __name__=='__main__':
    unittest.main()