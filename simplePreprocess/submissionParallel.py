# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 15:38:20 2015

@author: Fenno
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:46:08 2015

@author: fenno_000
"""
import RandomForestClassifier as learn
import readFeatureMatrix
import CreateSubmission
import os
import numpy as np
import zipfile

"""
Make it so every feature has a mean of 1, taken over the set of all drivers
"""
def normalizeFeatureMatrix(featureMatrix):
    numF = np.shape(featureMatrix)[0]
    for i in range(numF):
        meanF = np.mean(featureMatrix[i,:,:])
        featureMatrix[i,:,:] = featureMatrix[i,:,:] / meanF
    return featureMatrix

#make drivernrs.npy by calling 'readFeatureMatrix.getdrivernrs(featurespath)' and saving it
def getPage(driverNum):
    drivernrs = np.load('drivernrs.npy')
	return np.where(drivernrs == driverNum)[0][0]


"""
Differences with main file:
featureMatrix is npy file
driverNum is number of driver
"""
def main(featureMatrixPath, driverNum, outputSubmissionFolder, trainRealTrips = 200, trainFakeTrips = 200, normalize = False, digits = 5):

    features = np.load(featureMatrixPath) #problem with parallelism: constant reading of the same file
    numTrips = np.shape(features)[1]

    if normalize:
        features = normalizeFeatureMatrix(features)
		
    page = getPage(driverNum)
    
    trainTrips, trainLabels = learn.getTrips(features, page, trainRealTrips, trainFakeTrips)
    model = learn.trainModel(trainTrips, trainLabels, n_trees = 150, n_jobs = -1)
    tempprobs = learn.predictClass(model, np.transpose(featureMatrix[:,:,i]))
    probabilities = np.transpose(np.vstack((np.arange(1,numTrips+1), tempprobs)))
		
    CreateSubmission.appendProbabilities(outputSubmissionFolder + '/' + str(driverNum) + '.csv', driverNum, probabilities, fmtstring = '%0.10f'):
        
if __name__ == '__main__':

    from sys import argv
	
	outputFolder = 'submissionOutput'
	featureMatrixPath = 'featureMatrix.npy'
	
    driverNum = int(argv[1][:-4])
    main(featureMatrixPath, driverNum, outputFolder)