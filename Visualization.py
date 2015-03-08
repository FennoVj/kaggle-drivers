# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 14:00:06 2015

@author: Fenno
"""

from RandomForestClassifier import getTrips
from readFeatureMatrix import totalFeatureMatrix
import numpy as np
import matplotlib.pyplot as plt


def singleDriverFeature(featureMatrix, driverID, featureID, realtrips, faketrips, bins=50, proportional = True):
    proportion = float(realtrips) / float(faketrips)
    trips, labels = getTrips(featureMatrix, driverID, realtrips, faketrips)
    real = trips[np.where(labels)[0]]
    fake = trips[np.where(np.logical_not(labels))[0]]
    rhist, rbins = np.histogram([trip[featureID] for trip in real], bins)
    fhist, fbins = np.histogram([trip[featureID] for trip in fake], bins)
    if proportional:
        fhist = fhist * proportion    
    plt.plot(fbins[0:-1], fhist, 'r', rbins[0:-1], rhist, 'b')
    plt.show()
    
def allDriversFeatures(featureMatrix, featureID, bins = 100, percentile = 1):
    _, numTrips, numDrivers = np.shape(featureMatrix)
    page = np.reshape(featureMatrix[featureID], numTrips * numDrivers)
    minrange = np.percentile(page, percentile)
    maxrange = np.percentile(page, 100-percentile)
    hist, bins = np.histogram(page, bins, (minrange, maxrange))
    plt.plot(bins[0:-1], hist, 'b')
    #plt.axis([minrange, maxrange, 0, np.max(hist)])
    plt.show()

if __name__ == '__main__':
    
    dataPath = 'D:\\Documents\\Data\\MLiP'
    #featureMatrix = totalFeatureMatrix(dataPath + '\\features')
    featureMatrix = np.load(dataPath + '\\features.npy')
    numF, _, _ = np.shape(featureMatrix)
    #singleDriverFeature(featureMatrix, 0, 0, 200, 1000)
    for i in range(numF):
        allDriversFeatures(featureMatrix, i, 10000, 5)
    