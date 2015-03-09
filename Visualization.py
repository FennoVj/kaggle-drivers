# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 14:00:06 2015

@author: Fenno
"""

from RandomForestClassifier import getTrips
from readFeatureMatrix import totalFeatureMatrix
import numpy as np
import matplotlib.pyplot as plt


"""
Plots a single feature for some about of drivers
numdrivers: the number of drivers to plot the feature for, randomly chosen
featureID: what feature to plot from the matrix
realtrips: number of trips to take from each driver, suggest 200
faketrips: number of trips used to calculate the average over all drivers (in red), higher is better
"""
def singleDriverFeature(featureMatrix, numdrivers, featureID, realtrips=200, faketrips=10000, bins=50, percentile = 0):
    proportion = float(realtrips) / float(faketrips)
    
    _, numT, numD = np.shape(featureMatrix)
    page = np.reshape(featureMatrix[featureID], numT * numD)
    minrange = np.percentile(page, percentile)
    maxrange = np.percentile(page, 100-percentile)
    ftrips, _ = getTrips(featureMatrix, 0, 0, faketrips)
    fhist, fbins = np.histogram([trip[featureID] for trip in ftrips], bins, (minrange, maxrange))
    fhist = fhist * proportion    
    plt.plot(fbins[0:-1], fhist, 'r')    

    driverIDs = np.random.choice(np.arange(numD), numdrivers, False)
    for driver in driverIDs:
        trips, _ = getTrips(featureMatrix, driver, realtrips, 0)
        rhist, rbins = np.histogram([trip[featureID] for trip in trips], bins, (minrange, maxrange))
        plt.plot(rbins[0:-1], rhist)
    
    plt.show()

"""
Plots the average of all drivers for a given feature
allows cutting off percentile to prevent extremes
"""    
def allDriversFeatures(featureMatrix, featureID, bins = 100, percentile = 4):
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
    featureMatrix = np.load(dataPath + '\\featuresfourier.npy')
    numF, _, _ = np.shape(featureMatrix)
    singleDriverFeature(featureMatrix, 2, 0)
    #for i in range(numF):
    #    allDriversFeatures(featureMatrix, i, 10000, 5)
    