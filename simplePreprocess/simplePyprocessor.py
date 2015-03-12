#-*- coding: utf-8 -*-
"""
Created on Sat Mar 07 14:22:35 2015

@author: Fenno
"""

import numpy as np
#from scipy.interpolate import UnivariateSpline
from csv import reader
from os import listdir
from scipy.fftpack import fft


# All the drivers with all the trips (2736 x 200)
full_data = './data/drivers'
# Reduced training data set (547 x 40). At least 2 false trips for every driver plus the unknown ones.
# I kept the information which trips are false but that information should only be used for debugging
train_data = './data/train'

def get_tripfiles(folder):
    """
    returns a 2d-list (one list per driver) of the paths to all the trips
    in a given folder
    """
    drivers = ['%s/%s' % (folder, driver) for driver in listdir(folder)]
    trips = [['%s/%s' % (d, t) for t in listdir(d)] for d in drivers]
    return trips
    
def fftfeatures(feature, maxFeatures=20):
    y = fft(feature)
    return np.abs(y[0:maxFeatures])
	
def tripFixedLength(x, y, cumdist, datapoints = 1500):
    lperd = cumdist[-1] / float(datapoints-1)
    newx = np.zeros(datapoints)
    newy = np.zeros(datapoints)
    fill = 1
    for i in range(len(cumdist)):
        if cumdist[i] > fill * lperd:
            newx[fill] = x[i]
            newy[fill] = y[i]
            fill = fill + 1
    newx[-1] = x[-1]
    newy[-1] = y[-1]
    return newx, newy
    
#all_files = (get_tripfiles(full_data))

#training_files = array(get_tripfiles(train_data))

class trip(np.ndarray):

    def __new__(cls, filename, precision=1, **kwargs):
        with open(filename) as tripfile:
            head = tripfile.readline()
            trip = np.array(list(reader(tripfile)), dtype=float)
        return np.round(trip, decimals=precision).view(cls)

    def __init__(self, filename, **kwargs):

        #spline stuff
        #k = 3
        #s = 1
        X, Y = self.T
        self.n = self.shape[0] - 1
        self.t = np.arange(self.n)
        self.x = X
        self.y = Y
        self.dx = np.diff(self.x)
        self.dy = np.diff(self.y)        

        #The actual features
        self.v = np.hypot(self.dx, self.dy)
        self.v = np.hstack((self.v[0], self.v))
        self.o = np.arctan2(self.dy, self.dx)
        self.s = np.diff(self.o)
        self.s = np.hstack((self.s[0], self.s, self.s[-1]))
        self.a = np.diff(self.v)
        self.a = np.hstack((self.a[0], self.a))
        
        #polar coordinates
        self.rad = np.hypot(self.x, self.y)
        self.phi = np.arctan2(self.y, self.x)
        meanphi = np.mean(self.phi)
        self.normphi = self.phi - meanphi
        self.normX = self.rad * np.cos(self.normphi)
        self.normY = self.rad * np.sin(self.normphi)
        
        self.dist = np.copy(self.v)
        self.dist[self.dist > 50] = 0
        self.cumdist = np.cumsum(self.dist)
        
        #meanrad = np.mean(self.rad)
        self.newX, self.newY = tripFixedLength(self.normX, self.normY, self.cumdist)

if __name__ == '__main__':
#examples:
    total_time = lambda trip: trip.n
    total_distance = lambda trip: np.sum(np.hypot(np.diff(trip[:,0]), np.diff(trip[:,1])))
    straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
    straightness = lambda trip: straight_distance(trip) / total_distance(trip)
    sum_turnspeeds = lambda trip: np.sum(trip.s/trip.v)**2
    acceleration_to_dist = lambda trip: np.sum(trip.a**2)

    total_standstill_time = lambda trip: np.count_nonzero(trip.v < 0.1)

    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    tripp = trip(trippath)

    print(tripp.v)
    print(total_standstill_time(tripp))
