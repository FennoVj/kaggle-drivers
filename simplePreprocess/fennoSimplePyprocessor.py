#-*- coding: utf-8 -*-
"""
Created on Sat Mar 07 14:22:35 2015

@author: Fenno
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from csv import reader
from os import listdir


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


#all_files = (get_tripfiles(full_data))

#training_files = array(get_tripfiles(train_data))

class trip(np.ndarray):

    def __new__(cls, filename, precision=0, **kwargs):
        with open(filename) as tripfile:
            head = tripfile.readline()
            trip = np.array(list(reader(tripfile)), dtype=float)
        return np.round(trip, decimals=precision).view(cls)

    def __init__(self, filename, **kwargs):

        #spline stuff
        k = 3
        s = 1
        X, Y = self.T
        self.n = self.shape[0]
        self.t = np.arange(self.n)
        self.x = UnivariateSpline(self.t, X, k=k, s=s*self.n)
        self.y = UnivariateSpline(self.t, Y, k=k, s=s*self.n)
        self.dx = self.x.derivative(1)
        self.dy = self.y.derivative(1)
        self.ddx = self.x.derivative(2)
        self.ddy = self.y.derivative(2)
        self.dv = np.hypot(self.ddx(self.t), self.ddy(self.t))

        #The actual features
        self.v = np.hypot(self.dx(self.t), self.dy(self.t))
        self.o = np.arctan2(self.dy(self.t), self.dx(self.t))
        self.fv = lambda t: np.hypot(self.dx(t), self.dy(t))
        self.fo = lambda t: np.arctan2(self.dy(t), self.dx(t))
        self.s = self.fo(self.t+.5)-self.fo(self.t-.5)
        self.a = self.fv(self.t+.5)-self.fv(self.t-.5)
        
        #polar coordinates
        self.rad = np.hypot(self.x(self.t), self.y(self.t))
        self.phi = np.arctan2(self.y(self.t), self.x(self.t))
        meanphi = np.mean(self.phi)
        self.normphi = self.phi - meanphi
        self.normX = self.rad * np.cos(self.normphi)
        self.normY = self.rad * np.sin(self.normphi)



if __name__ == '__main__':
#examples:
    total_time = lambda trip: trip.n
    total_distance = lambda trip: np.sum(hypot(np.diff(trip[:,0]), np.diff(trip[:,1])))
    straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
    straightness = lambda trip: straight_distance(trip) / total_distance(trip)
    sum_turnspeeds = lambda trip: np.sum(trip.s/trip.v)**2
    acceleration_to_dist = lambda trip: np.sum(trip.a**2)

    total_standstill_time = lambda trip: np.count_nonzero(trip.v < 0.1)

    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    tripp = trip(trippath)

    print(tripp.v)
    print(total_standstill_time(tripp))
