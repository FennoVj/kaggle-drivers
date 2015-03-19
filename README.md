# Machine Learning in Practice - Team Mavericks
## Driver Telematics

url: https://gitlab.science.ru.nl/kaggle-hacking/kaggle-drivers

This is an entry for the driver telematics competition on kaggle:
http://www.kaggle.com/c/axa-driver-telematics-analysis

The members of Team Mavericks are:
* Andra Alexa
* Hue Dang
* Jona Heidsick
* Fenno Vermeij

This code is available for educational use only, prepare to get your hands dirty if you use it!

File overview:
* CreateSubmission.py: Creating the output file
* RandomForestClassifier.py: Learning/applying model, as well as testing the model
* Visualization.py: Some methods to visualize the features
* makeSubmissionFromFeatures.py: Method to take in the features and create a submission
* readFeatureMatrix: Reading the features
* simplePreprocess/DriverFeatures.py: Creating the driverfeatures
* simplePreprocess/extractSimpleFeatures.py: With raw data as input, create the feature matrices
* simplePreprocess/simpleFeatures.py: Creating the simple features
* simplePreprocess/simplePyprocessor.py: Class for storing information about the trips

To use this code to make a submission, first you need to create the featurematrices.
This is done by running the 'extractSimpleFeatures.py' with a number as input.
For example: 'extractSimpleFeatures.py 1' will create the feature matrix of the first driver. The runtime for calculating every feature is very long, but can be parallelized. It took us 2-3 hours on a machine with 24 cores.
Once all 2736 feature matrices have been created, you can make a submission,
by running 'makeSubmissionFromFeatures.py'. Depending on the foldsize, this can a long time. Our final submission was made in about 10 hours, with a foldsize of 10.
