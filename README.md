# Image-Colorization

## Loading the dataset:
* Open the terminal and run the following command:

* python data_preprocessing.py <path_to_images_folder> <name_of_annotations_file.csv>

## Regressor:

Python regressor.py <path_to_images_folder> <name_of_annotations_file.csv>

## Colorization:

* python Colorizer.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

** Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice

* 4.GPU Computing:

python GPU_computing.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice

* 5.Tuning:

python Hyperparameter_tuning.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice


python Transfer_learning.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice
