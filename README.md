# Image-Colorization
How to run the files:

1.Loading the dataset:
Open the terminal and run the following command:

python data_preprocessing.py <path_to_images_folder> <name_of_annotations_file.csv>

2.Regressor:

Python regressor.py <path_to_images_folder> <name_of_annotations_file.csv>

3.Colorization:

python Colorizer.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice

4.GPU Computing:

python GPU_computing.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice

5.Tuning:

python Hyperparameter_tuning.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice


python Transfer_learning.py <path_to_images_folder> <name_of_trainingdata_file.csv> <name_of_testingdatafile.csv>

Note: <name_of_trainingdata_file.csv>, <name_of_testingdata_file.csv> can be named as user’s choice
![image](https://github.com/rev97/Image-Colorization/assets/89413538/4475e631-cce3-4b97-a476-0936f090cfe4)
