"""
Created on Sat Mar 07 13:05:04 2015

@author: Fenno
"""
import numpy as np

from simplePyprocessor import trip

"""
Threshold Feature
Takes only elements that exceed threshold, and does an operation on them to get final feature
"""
def thF(array, operation = np.mean, threshold = 0, greater = True):
    values = array[(array > threshold)] if greater else array[(array < threshold)]
    return operation(values) 
"""
Count Feature
Counts all nonzero (or true) elements, and returns the number of them
by default, normalizes it, so it returns the proportion of elements that are nonzero (or true)
"""
def coF(array, normalize = True):
    count = float(np.count_nonzero(array))
    return count / len(array) if normalize else count

#Used features: n, s, t, v, a

#np.seterr(divide = 'ignore')

#Basic Features
total_time = lambda trip: trip.n
total_distance = lambda trip: np.sum(np.hypot(np.diff(trip[:,0]), np.diff(trip[:,1])))
straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
straightness = lambda trip: np.divide(float(straight_distance(trip)), float(total_distance(trip)))

sum_turnspeeds = lambda trip: np.sum(trip.s/trip.v)
acceleration_to_dist = lambda trip: np.sum(trip.a**2)

#Hue's ideas + enhancements by Fenno
mean_acceleration = lambda threshold: lambda trip: thF(trip.a, np.mean, threshold, True)#sample threshold: 0
mean_decceleration = lambda threshold: lambda trip: thF(trip.a, np.mean, threshold, False) #sample threshold: 0
total_standstill_time = lambda threshold: lambda trip: coF(trip.v < threshold, False) #sample threshold: 0.1
proportion_standstill_time =  lambda threshold: lambda trip: coF(trip.v < threshold, True) 

#Fenno's ideas
turnspeed_velocity = lambda trip: np.sum(trip.s * trip.v)
mean_turnspeed_velocity = lambda threshold: lambda trip: thF(trip.s * trip.v, np.mean, threshold, True)#sample threshold: 0
turnspeed_acceleration = lambda trip: np.sum(trip.s * trip.a)
sum_turnacc = lambda trip : np.sum(trip.s / trip.a)
mean_turnacc = lambda threshold: lambda trip: thF(trip.s * trip.a, np.mean, threshold, True)#sample threshold: 0
mean_steering_right = lambda threshold: lambda trip: thF(trip.s, np.mean, threshold, True)#sample threshold: 0
mean_steering_left = lambda threshold: lambda trip: thF(trip.s, np.mean, threshold, False)#sample threshold: 0
number_acc_threshold = lambda threshold: lambda trip: coF(trip.a > threshold)#sample threshold: 0.2
number_dec_threshold = lambda threshold: lambda trip: coF(trip.a < threshold)#sample threshold: 0.2
number_steering_threshold = lambda threshold: lambda trip: coF(np.abs(trip.s) > threshold) #sample threshold: 0.05

#Minima and maxima
max_velocity = lambda trip: np.max(trip.v)
min_velocity = lambda trip: np.min(trip.v)
max_acceleration = lambda trip: np.max(trip.a)
min_acceleration = lambda trip: np.min(trip.a)
max_steering = lambda trip: np.max(trip.s)
min_steering = lambda trip: np.min(trip.s)

#new
mean_steering = lambda trip: np.mean(trip.s)
mean_acceleration_total = lambda trip: np.mean(trip.a)
mean_velocity = lambda trip: np.mean(trip.v)
mean_velocity_th = lambda threshold: lambda trip: thF(trip.v, np.mean, threshold, True)
mean_acceleration_th = lambda threshold: lambda trip: thF(trip.a, np.mean, threshold, True)

std_velocity = lambda trip: np.std(trip.v)
std_acceleration = lambda trip: np.std(trip.a)
std_steering = lambda trip: np.std(trip.s)

#polar stuff
mean_rad = lambda trip: np.mean(trip.rad)
std_rad = lambda trip: np.std(trip.rad)
mean_x = lambda trip: np.mean(trip.normX)
mean_y = lambda trip: np.mean(trip.normY)
std_x = lambda trip: np.std(trip.normX)
std_y = lambda trip: np.std(trip.normY)
std_phi = lambda trip: np.std(trip.normphi)

features = [total_time, total_distance, straight_distance, straightness, acceleration_to_dist, \
mean_acceleration(0), mean_decceleration(0), total_standstill_time(0.1), turnspeed_velocity, turnspeed_acceleration,\
mean_turnacc(0), mean_steering_right(0), mean_steering_left(0), number_acc_threshold(0.2), \
number_dec_threshold(0.2), number_steering_threshold(0.05), max_velocity, min_velocity, max_acceleration, min_acceleration, \
max_steering, min_steering, mean_steering, mean_acceleration_total, mean_velocity, \
std_velocity, std_acceleration, std_steering, mean_rad, std_rad, mean_x, mean_y, std_x, std_y, std_phi]
#sum_turnspeeds, sum_turnacc left out

#features, leaving out some of the most contrived ones, to see if better performance
#features = [total_time, total_distance, straight_distance, straightness, mean_acceleration_total, \
#mean_velocity, mean_velocity_th(10), mean_acceleration_th(0.2), max_velocity, max_acceleration, max_steering, min_steering, \
#mean_acceleration(0), mean_decceleration(0), mean_steering, mean_turnspeed_velocity(0), total_standstill_time(0.1)]



if __name__=='__main__':
    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    trip = trip(trippath)
    samplefeatures = [total_distance,  total_standstill_time(0.1)]
    output = [float(f(trip)) for f in samplefeatures]
    print(output)
    #print(np.sort(trip.a))