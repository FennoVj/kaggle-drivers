"""
Created on Sat Mar 07 13:05:04 2015

@author: Fenno
"""
import numpy as np

#from simplePyprocessor import trip

"""
Threshold Feature
Takes only elements that exceed threshold, and does an operation on them to get final feature
"""
def thF(array, operation = np.mean, threshold = 0, greater = True):
    values = array[(array > threshold)] if greater else array[(array < threshold)]
    if len(values) == 0:
        values = np.array([0])
    return operation(values) 
"""
Count Feature
Counts all nonzero (or true) elements, and returns the number of them
by default, normalizes it, so it returns the proportion of elements that are nonzero (or true)
"""
def coF(array, normalize = True):
    count = float(np.count_nonzero(array))
    return count / len(array) if normalize else count
    
"""
Makes it so you can safely do operations like np.max, np.min, np.mean, etc, 
when you don't know if the array is empty or not.
"""
def safe(array, operation, default = 0):
    if len(array) == 0:
        return default
    return operation(array)
    
"""
Makes array of the lengths of streaks of True
example: [True, True, True, False, False, True, True, False] would return [3,2]
input array has to be boolean
you can also count False streaks by setting value argument
"""
def countStreak(array, value=True):
    firstval = array[0]
    streaklengths = np.diff(np.where(np.hstack((True, np.diff(array), True))))[0]
    if firstval == value:
        return streaklengths[::2]
    return streaklengths[1::2]
    
def lengthOfStreaks(array, cumdist, value=True, endCond = None):
    if endCond is None:
        endCond = np.ones(len(array))
    edges = np.where(np.diff(array))[0]
    s = 0 if array[0] else []
    e = len(array)-1 if array[-1] else []
    edges = np.hstack((s, edges, e))
    startseg = 0
    counter = 0
    result = np.zeros(len(edges)/2)
    for i, e in enumerate(edges):
        if i%2 == 0:
            startseg = cumdist[int(e)+1]
        elif endCond[int(e)]:
            result[counter] = cumdist[int(e)] - startseg
            counter = counter + 1
    return result[:counter]
	
#Used features: n, s, t, v, a

#np.seterr(divide = 'ignore')

#Basic Features
total_time = lambda trip: trip.n
total_distance = lambda trip: np.sum(trip.dist)
straight_distance = lambda trip: np.hypot(trip[-1,0], trip[-1,1])
straightness = lambda trip: np.divide(float(straight_distance(trip)), float(total_distance(trip)))

sum_turnspeeds = lambda trip: np.sum(trip.s/trip.v)
acceleration_to_dist = lambda trip: np.sum(trip.a**2)

dist_excl_hyperjump = lambda maxdist: lambda trip: float(np.sum(trip.dist[trip.dist < maxdist])) #sample: 80

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

#means +std because obvious
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


#Hue new features
#proportion_constant_speed_time  sample threshold = 1 , corresponding to 3.6 km/h different
proportion_constant_speed_time = lambda threshold: lambda trip: coF(np.abs(trip.a) <= threshold, True)   

proportion_deceleration_time = lambda trip: coF(trip.a < 0, True)
proportion_acceleration_time = lambda trip: coF(trip.a > 0, True)
mean_velocity_excluding_stop = lambda threshold: lambda trip: sum(trip.v) / coF(trip.v > threshold, False) #sample threshold: 0.1

max_product_velocity_acceleration = lambda trip: safe(trip.a[(trip.a>0)] * trip.v[(trip.a>0)], np.max)
min_product_velocity_acceleration = lambda trip: safe(trip.a[(trip.a>0)] * trip.v[(trip.a>0)], np.min)
mean_product_velocity_acceleration = lambda trip: safe(trip.a[(trip.a>0)] * trip.v[(trip.a>0)], np.mean)
std_product_velocity_acceleration = lambda trip: safe(trip.a[(trip.a>0)] * trip.v[(trip.a>0)], np.std)
min_product_velocity_deceleration = lambda trip: safe(trip.a[(trip.a<0)] * trip.v[(trip.a<0)], np.min)
max_product_velocity_deceleration = lambda trip: safe(trip.a[(trip.a<0)] * trip.v[(trip.a<0)], np.max)
mean_product_velocity_deceleration = lambda trip: safe(trip.a[(trip.a<0)] * trip.v[(trip.a<0)], np.mean)
std_product_velocity_deceleration = lambda trip: safe(trip.a[(trip.a<0)] * trip.v[(trip.a<0)], np.std)

#% of time in speed interval  [a, b]  13.8 m/s ~ 50 km/h, [19.4  70]  [33.3  120]
proportion_speed_in_interval = lambda a,b: lambda trip: coF((trip.v>=a) & (trip.v<=b), True)
proportion_acceleration_in_interval = lambda a,b: lambda trip: coF((trip.a>=a) & (trip.a<=b), True)

#Jonah's idea: use percentiles, quartile distances, and medians
median_velocity = lambda trip: np.median(trip.v)
median_acceleration_total = lambda trip: np.median(trip.a)
median_acceleration = lambda threshold: lambda trip: thF(trip.a, np.median, threshold, True)
median_decceleration = lambda threshold: lambda trip: thF(trip.a, np.median, threshold, False)
velocity_percentile = lambda percentile: lambda trip: np.percentile(trip.v, percentile)
acceleration_percentile = lambda percentile: lambda trip: np.percentile(trip.a, percentile)
steering_percentile = lambda percentile: lambda trip: np.percentile(trip.s, percentile)
velocity_quartdist = lambda trip: np.percentile(trip.v, 75) - np.percentile(trip.v, 25)
steering_quartdist = lambda trip: np.percentile(trip.a, 75) - np.percentile(trip.s, 25)
acceleration_quartdist = lambda trip: np.percentile(trip.s, 75) - np.percentile(trip.a, 25)

#FourierTransforms
fourierAcc = lambda coeff: lambda trip: trip.fouriera[coeff]
fourierVec = lambda coeff: lambda trip: trip.fourierv[coeff]

#Counting the number of stops/acceleration/decceleration events
number_stops = lambda threshold: lambda trip: len(countStreak(trip.v < threshold)) #sample threshold: 0.1
number_accel = lambda threshold: lambda trip: len(countStreak(trip.a > threshold)) #sample: 0.2
number_decel = lambda threshold: lambda trip: len(countStreak(trip.a < threshold))
num_accel_meter = lambda threshold: lambda trip: np.divide(len(countStreak(trip.a > threshold)), dist_excl_hyperjump(80)(trip))
num_decel_meter = lambda threshold: lambda trip: np.divide(len(countStreak(trip.a < threshold)), dist_excl_hyperjump(80)(trip))
num_stop_meter = lambda threshold: lambda trip: np.divide(len(countStreak(trip.v < threshold)), dist_excl_hyperjump(80)(trip))

#length-of-streak-related features
distance_of_stops = lambda threshold: lambda trip: safe(lengthOfStreaks(trip.v < threshold, trip.cumdist), np.mean)
distance_of_subtrips = lambda threshold: lambda trip: safe(lengthOfStreaks(trip.v > threshold, trip.cumdist), np.mean)
distance_until_stop = lambda threshold: lambda trip: safe(lengthOfStreaks(trip.a < 0, trip.cumdist, endCond = trip.v<threshold), np.mean) 

total_raw_distance = lambda trip: np.sum( trip.v )
std_acceleration = lambda trip: safe(trip.a[(trip.a>0)], np.std)
std_deceleration = lambda trip: safe(trip.a[(trip.a<0)], np.std)
num_acceleration_deceleration_changes = lambda trip: np.count_nonzero(np.diff(trip.a > 0))

features = [total_time,
                total_distance, 
			straight_distance, 
			straightness, 
			acceleration_to_dist, 
			mean_acceleration(0), 
			mean_decceleration(0), 
			total_standstill_time(0.1), 
			turnspeed_velocity, 
			turnspeed_acceleration,
			mean_turnacc(0), 
			mean_steering_right(0), 
			mean_steering_left(0), 
			number_acc_threshold(0.2), 
			number_dec_threshold(0.2), 
			number_steering_threshold(0.05), 
			max_velocity, 
			min_velocity, 
			max_acceleration, 
			min_acceleration, 
			max_steering,
			min_steering, 
			mean_steering, 
			mean_acceleration_total, 
			mean_velocity, 
			std_velocity, 
			std_acceleration, 
			std_steering, 
			mean_rad, 
			std_rad, 
			mean_x, 
			mean_y, 
			std_x, 
			std_y, 
			std_phi,
			proportion_constant_speed_time(1),
			proportion_deceleration_time, 
			proportion_acceleration_time, 
			max_product_velocity_acceleration,  
			min_product_velocity_acceleration, 
			mean_product_velocity_acceleration,  
			std_product_velocity_acceleration,  
			max_product_velocity_deceleration, 
			min_product_velocity_deceleration,  
			mean_product_velocity_deceleration,  
			std_product_velocity_deceleration, 
			proportion_speed_in_interval(13.8, 19.4), 
			proportion_acceleration_in_interval(0.05,0.1), 
			median_velocity, 
			median_acceleration_total, 
			median_acceleration(0), 
			median_decceleration(0), 
			velocity_percentile(5), 
			velocity_percentile(25),
			velocity_percentile(75), 
			velocity_percentile(95), 
			acceleration_percentile(5), 
			acceleration_percentile(25), 
			acceleration_percentile(75),
			acceleration_percentile(95), 
			steering_percentile(5), 
			steering_percentile(25), 
			steering_percentile(75), 
			steering_percentile(95), 
			velocity_quartdist, 
			steering_quartdist, 
			acceleration_quartdist,
			dist_excl_hyperjump(80), 
			number_stops(0.1), 
			number_accel(0.5), 
			number_decel(-0.5), 
			num_accel_meter(0.5), 
			num_decel_meter(-0.5), 
			num_stop_meter(0.1),
               distance_of_stops(1),
               distance_of_subtrips(1),
               distance_until_stop(1),
                total_standstill_time(1),
                number_acc_threshold(0.5),
                number_dec_threshold(0.5),
    		      proportion_speed_in_interval(0, 1), 
                proportion_speed_in_interval(0, 4.16), 
                proportion_speed_in_interval(4.16,8.333), 
                proportion_speed_in_interval(8.333,13.89),
                proportion_speed_in_interval(13.89, 19.44), 
                proportion_speed_in_interval(19.44, 25), 
                proportion_speed_in_interval(25, 30.55), 
                proportion_speed_in_interval(30.55,1000),
                proportion_acceleration_in_interval(-2.777, -0.69444),
                proportion_acceleration_in_interval(-0.69444, -0.41666),
                proportion_acceleration_in_interval(-0.41666, -0.27777),
                proportion_acceleration_in_interval(-0.27777, -0.13888),
                proportion_acceleration_in_interval(-0.13888, 0),
                proportion_acceleration_in_interval(0, 0.13888),
                proportion_acceleration_in_interval(0.13888, 0.27777),
                proportion_acceleration_in_interval(0.27777, 0.41666),
                proportion_acceleration_in_interval(0.41666, 0.69444),
                proportion_acceleration_in_interval(0.69444, 2.77777),
                number_stops(3), 
                number_accel(0.3), 
                number_decel(-0.3), 
                num_accel_meter(0.3), 
                num_decel_meter(-0.3), 
                num_stop_meter(2),
                distance_of_stops(0.1),
                distance_of_subtrips(5),
                distance_until_stop(0.1)]
#sum_turnspeeds, sum_turnacc, mean_velocity_excluding_stop left out because of exessive zero division




if __name__=='__main__':
    #trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'
    #trip = trip(trippath)
    #samplefeatures = [total_distance,  total_standstill_time(0.1)]
    #output = [float(f(trip)) for f in samplefeatures]
    #print(output)
    print(len(features))
    #print dist_excl_hyperjump(80)(trip)
    #print(np.sort(trip.a))

    x = np.array([False, True, True, True, False, True])
    cumdist = np.array([0,2,3,4,5,6])
    print(lengthOfStreaks(x, cumdist))