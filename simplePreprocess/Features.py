# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 13:05:04 2015

@author: Fenno
"""
import numpy as np

from pyprocessor import trip

"""
Threshold Feature
Takes only elements that exceed threshold, and does an operation on them to get final feature
"""
def thF(array, operation = np.mean, threshold = 0, greater = True):
    values = (array > threshold) if greater else (array < threshold)    
    return operation(values)
    
"""
Count Feature
Counts all nonzero (or true) elements, and returns the number of them
by default, normalizes it, so it returns the proportion of elements that are nonzero (or true)
"""
def coF(array, normalize = True):
    count = np.count_nonzero(array)
    return float(count) / len(array) if normalize else count

#Used features: n, s, t, v, a

np.seterr(divide = 'ignore')

#Basic Features
total_time = lambda trip: trip.n
total_distance = lambda trip: np.sum(np.hypot(np.diff(trip[:,0]), np.diff(trip[:,1])))
straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
straightness = lambda trip: straight_distance(trip) / total_distance(trip)

sum_turnspeeds = lambda trip: np.sum(trip.s(trip.t)/trip.v(trip.t))
acceleration_to_dist = lambda trip: np.sum(trip.a(trip.t)**2)

#Hue's ideas + enhancements by Fenno
mean_acceleration = lambda threshold: lambda trip: thF(trip.a(trip.t), np.mean, threshold, True)#sample threshold: 0
mean_decceleration = lambda threshold: lambda trip: thF(trip.a(trip.t), np.mean, threshold, False) #sample threshold: 0
total_standstill_time = lambda threshold: lambda trip: coF(trip.v(trip.t) < threshold) #sample threshold: 0.1

#Fenno's ideas
turnspeed_velocity = lambda trip: np.sum(trip.s(trip.t) * trip.v(trip.t))
mean_turnspeed_velocity = lambda threshold: lambda trip: thF(trip.s(trip.t) * trip.v(trip.t), np.mean, threshold, True)#sample threshold: 0
turnspeed_acceleration = lambda trip: np.sum(trip.s(trip.t) * trip.a(trip.t))
sum_turnacc = lambda trip : np.sum(trip.s(trip.t) / trip.a(trip.t))
mean_turnacc = lambda threshold: lambda trip: thF(trip.s(trip.t) * trip.a(trip.t), np.mean, threshold, True)#sample threshold: 0
mean_steering_right = lambda threshold: lambda trip: thF(trip.s(trip.t), np.mean, threshold, True)#sample threshold: 0
mean_steering_left = lambda threshold: lambda trip: thF(trip.s(trip.t), np.mean, threshold, False)#sample threshold: 0
number_acc_threshold = lambda threshold: lambda trip: coF(trip.a(trip.t) > threshold)#sample threshold: 0.2
number_dec_threshold = lambda threshold: lambda trip: coF(trip.a(trip.t) < threshold)#sample threshold: 0.2
number_steering_threshold = lambda threshold: lambda trip: coF(np.abs(trip.s(trip.t)) > threshold) #sample threshold: 0.05

#Minima and maxima
max_velocity = lambda trip: np.max(trip.v(trip.t))
min_velocity = lambda trip: np.min(trip.v(trip.t))
max_acceleration = lambda trip: np.max(trip.a(trip.t))
min_acceleration = lambda trip: np.min(trip.a(trip.t))
max_steering = lambda trip: np.max(trip.s(trip.t))
min_stering = lambda trip: np.min(trip.s(trip.t))

features = [total_time, total_distance, straight_distance, straightness, sum_turnspeeds, acceleration_to_dist, \
mean_acceleration(0), mean_decceleration(0), total_standstill_time(0.1), turnspeed_velocity, turnspeed_acceleration,\
sum_turnacc, mean_turnacc(0), mean_steering_right(0), mean_steering_left(0), number_acc_threshold(0.2), \
number_dec_threshold(0.2), number_steering_threshold(0.05)]



if __name__=='__main__':
    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    trip = trip(trippath)    
    samplefeatures = [total_distance,  total_standstill_time(0.1)]
    output = [f(trip) for f in samplefeatures]
    print(output)
    #print(np.sort(trip.a(trip.t)))