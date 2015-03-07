# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:12:33 2015

@author: vermeij
"""
import os
import numpy as np

"""Sorts [1.csv, 10.csv, 2.csv, ...] as [1.csv, 2.csv, ...]"""
def sortNumerical(filelist):
    filelist = [int(f[:-4]) for f in filelist]
    return [`f` + '.csv' for f in sorted(filelist)]
    
def getdrivernrs(tdatpath):
    files = os.listdir(tdatpath)
    files = sortNumerical(files)
    return np.array([int(f[:-4]) for f in files])

def makeFeatureMatrix(tdatpath, numFeatures, numTrips, numDrivers, fixnan = True):
    featurematrix = np.zeros((numFeatures, numTrips,numDrivers))
    files = os.listdir(tdatpath)
    files = sortNumerical(files)
    for i in range(numDrivers):
        csvpath = os.path.join(tdatpath, files[i])
        features = np.transpose(np.genfromtxt(csvpath, dtype = 'float', delimiter = ','))
        if fixnan:
            features = np.nan_to_num(features)
        featurematrix[:,:,i] = features
    return featurematrix   

#number of ',' in first line + 1 
def getNumFeatures(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        return first_line.count(',') + 1
   
#number of lines in file
def getNumTrips(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
    
def printMatlabStyle(threedmatrix):
    for i in range(np.shape(threedmatrix)[2]):
        print 'matrix[:,:,' + `i` + '] = '
        print threedmatrix[:,:,i]

if __name__ == '__main__':
    print np.__version__    
    tdatpath = 'D:\Documents\Data\MLiP\output'
    numFeatures = 6
    numTrips = 200
    numDrivers = 10
    printMatlabStyle(makeFeatureMatrix(tdatpath, numFeatures, numTrips, 10))