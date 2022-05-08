# ImageClassificationCompressionTechniques
This project is used for COMS 6998 Deep learning practical sys class. The owners of this repository are Sachin Devashyam and Vibhas Naik

# Cloud Setup
1) Install kaggle.

    a)pip install kaggle
  
2) Setup kaggle credentials in VM by following this [link](https://adityashrm21.github.io/Setting-Up-Kaggle/)

3) Download dataset.

    a)kaggle datasets download -d hgunraj/covidxc
    
# Project Setup

1) create a directory called as project in the root folder.
2) Clone "ImageClassificationCompressionTechniques" into the project folder.
3) Run the following 3 commands.
    - mkdir project/ImageClassificationCompressionTechniques/Data/CovidDataSet
    - mv 2A_images/ project/ImageClassificationCompressionTechniques/Data/CovidDataSet
    - mv metadata.csv project/ImageClassificationCompressionTechniques/Data/CovidDataSet
    - mv *.txt project/ImageClassificationCompressionTechniques/Data/CovidDataSet

# Commands to run training script

1) main.py runs ResNet20 training script and stores the trained model in Results/ResNet18/ResNet_{epoch}.h5 (Please ignore ResNet18 folder name. This was due to a confusion)
2) main_resnet_50.py runs ResNet50 training script and stores the trained model in Results/ResNet50/ResNet_{epoch}.h5

# Commands to run compression script

1) main_pruned.py and main_clustered.py runs pruning and compression training scripts for the models save in Results/ResNet{18 or 50} folder. The compressed models are saved in the same Results folder with either 'Pruned' or 'Cluster' added as a suffix.

# Commands to run test accuracy script

1) get_test_accuracy.py evaluates the test accuracy of the models generated. The script must be modified to point to the correct model.

# Convert to Tensorflow.js format
- In order to convert a model from a normal Keras model to a tfjs compatible format, call `main_convert_tfjs.py input_file output_dir` where `input_file` is the path to the keras model you want to convert and `output_dir` is the directory you want to output the converted model.json and its weight binaries to

# Results

![ResNet20ModelTestAccuracy](Results/Graphs/ResNet20ModelAccuracy.jpg?raw=true "Resnet20 Model Test Accuracy")
