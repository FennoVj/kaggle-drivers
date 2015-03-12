# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:46:08 2015

@author: fenno_000
"""
import RandomForestClassifier as learn
#import GradientBoostedClassifier as learn
import readFeatureMatrix
import CreateSubmission
import numpy as np
import zipfile
from sklearn.decomposition import PCA
"""
Make it so every feature has a mean of 1, taken over the set of all drivers
"""
def normalizeFeatureMatrix(featureMatrix):
    numF = np.shape(featureMatrix)[0]
    for i in range(numF):
        meanF = np.mean(featureMatrix[i,:,:])
        featureMatrix[i,:,:] = featureMatrix[i,:,:] / meanF
    return featureMatrix    
    
def pca(featureMatrix, pcpCompNumber):
    numTrips = np.shape(featureMatrix)[1]
    numDrivers = np.shape(featureMatrix)[2]
    newFeatureMatrix = np.zeros((pcpCompNumber, numTrips, numDrivers))
    for i in range(numDrivers):
        pca = PCA(n_components=pcpCompNumber)
        pca.fit(featureMatrix[:,:,i])
        #print(pca.explained_variance_ratio_)
        newFeatureMatrix[:,:,i]=pca.components_        
    return newFeatureMatrix

"""
Reads the featurematrix, does the machine learning, creates models, uses them to predict every trip, makes submission file
Note: you can change the logistic regression parameter by changing the lg.trainModel line.
I didn't include this in the function parameters because it already had a lot of function parameters,
but feel free to change that line and see what happens to the results
"""
def makeSubmissionScript(featureMatrixPath, outputSubmissionPath, trainRealTrips = 200, trainFakeTrips = 200, normalize = False, digits = 5):
    #Read Feature Matrix    
    #featureMatrix = readFeatureMatrix.totalFeatureMatrix(featureMatrixPath)

    #ShortCut
    featureMatrix = np.load('D:\\Documents\\Data\\MLiP\\featurestotal.npy')
    #np.save('D:\\Documents\\Data\\MLiP\\featurestotal', featureMatrix)
    
    print(np.shape(featureMatrix))
    featureMatrix = np.delete(featureMatrix, [4,8,9,14,15,38,39,40,41,42,43,44,45], 0 )
    
    drivernrs = readFeatureMatrix.getdrivernrs(featureMatrixPath)
    print('Done Reading Feature matrix!')
    print(np.shape(featureMatrix))
    
    numFeat, numTrips, numDrivers = np.shape(featureMatrix)
    
    numTrips = 200
    #readFeatureMatrix.printMatlabStyle(featureMatrix)
    importances = np.zeros((numFeat, numDrivers))
    
    #Train and immediately predict all trips from a single driver, one by one
    probabilities = np.zeros((numTrips, 2, numDrivers))
    for i in range(numDrivers):
        #trainTrips, trainLabels, pathlist = learn.getTrips(featureMatrix, i, trainRealTrips, trainFakeTrips)
        trainTrips = np.transpose(featureMatrix[:,:,i])
        realTrips = trainTrips[:numTrips,:]
        trainLabels = np.hstack((np.ones(trainRealTrips), np.zeros(trainFakeTrips)))

        #model = learn.trainModel(trainTrips, trainLabels) #Add other parameters here to test
        model = learn.trainModel(trainTrips, trainLabels, n_trees = 200, n_jobs = -1)
        
        importances[:,i] = model.feature_importances_
        
        tempprobs = learn.predictClass(model, realTrips)
        
        #avgweight = np.sum(tempprobs)
        #sampleweight = np.hstack((tempprobs, avgweight * np.ones(numTrips)))
        #model = learn.trainModel(trainTrips, trainLabels, n_trees = 200, n_jobs = -1, sample_weight = sampleweight)
        #tempprobs = learn.predictClass(model, realTrips)

        probabilities[:,:,i] = np.transpose(np.vstack((np.arange(1,numTrips+1), tempprobs)))
        if i%10 == 0:
            print("Done learning driver " + `i`)
    print('Done calculating probabilities!')
    #readFeatureMatrix.printMatlabStyle(probabilities)
    
    print(np.mean(importances, axis=1))    
    
    #Makes submission file
    fmtstring = '%0.' + `digits` + 'f'
    CreateSubmission.createSubmissionfileFrom3D(outputSubmissionPath, probabilities, drivernrs = drivernrs, fmtstring = fmtstring)
        
if __name__ == '__main__':
    #print(np.vstack((np.arange(5), np.array([0.3,0.2,0.1,0.8,0.9]))))
    
    featureMatrixPath = 'D:\\Documents\\Data\\MLiP\\features'
    outputSubmissionPath = 'D:\\Documents\\Data\\MLiP\\submission.csv'
    trainRealTrips = 200
    trainFakeTrips = 200 #Change to 200 for real thing
    normalize = False
    significantdigits = 5
    makeSubmissionScript(featureMatrixPath, outputSubmissionPath, trainRealTrips, trainFakeTrips, normalize, significantdigits)
    
    #zip the submission, makes it ~3x smaller
    zf = zipfile.ZipFile(outputSubmissionPath[:-4] + '.zip', mode='w')
    zf.write(outputSubmissionPath, 'submission.csv', compress_type=zipfile.ZIP_DEFLATED)
    zf.close()
    print('Done creating submission!')