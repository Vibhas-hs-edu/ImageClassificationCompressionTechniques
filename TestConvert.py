from Models.Convert import Convert
import unittest
import tensorflow as tf

RES20_FILE = 'res20.h5'
RES50_FILE = 'Results/ResNet50/ResNet50_3.h5'



class ConvertToJs(unittest.TestCase):

    @unittest.skip
    def test1(self):
        
        model = tf.keras.models.load_model(RES20_FILE)

        convert = Convert(model)

        convert.to_tfjs('JS/model')
    
    def test3(self):

        model = tf.keras.models.load_model(RES50_FILE)



        # model.summary()

        convert = Convert(model)

        print('Converting now')

        convert.to_tfjs('JS/model/Res50Test')

# Run the tests
if __name__=='__main__':
    unittest.main()