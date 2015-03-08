import numpy as np

from scipy.interpolate import UnivariateSpline
from matplotlib.pylab import plot
from numpy import array, arange, abs, exp, hypot, arctan2, sum
from functools import partial
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
            trip = array(list(reader(tripfile)), dtype=float)
        return np.round(trip, decimals=precision).view(cls)

    def __init__(self, filename, **kwargs):
        X, Y = self.T #T is transpose
        #self.driver = filename.split('/')[-2]
        self.n = self.shape[0]
        self.t = arange(self.n)
        self.distance_driven = sum(hypot(*(self[1:] - self[:-1]).T))
        self.distance_straight = hypot(X[-1], Y[-1])
        self.straightness = self.distance_straight / self.distance_driven
        self.__interpolators__(**kwargs)

    def __interpolators__(self, k=3, s=1):
        X, Y = self.T
        self.x = UnivariateSpline(self.t, X, k=k, s=s*self.n)
        self.y = UnivariateSpline(self.t, Y, k=k, s=s*self.n)
        #radius
        self.rad = lambda t: hypot(self.x(t), self.y(t))
        #angle (from origin)
        self.phi = lambda t: arctan2(self.y(t), self.x(t))
        #self.p = lambda t: (self.r(t), self.a(t))
        self.X = self.x.antiderivative
        self.Y = self.y.antiderivative
        #self.P = lambda t: (self.X(t), self.Y(t))
        self.dx = self.x.derivative(1)
        self.dy = self.y.derivative(1)
        #v is velocity
        self.v = lambda t: hypot(self.dx(t), self.dy(t))
        #o is orientation
        self.o = lambda t: arctan2(self.dy(t), self.dx(t))
        self.ddx = self.x.derivative(2)
        self.ddy = self.y.derivative(2)
        self.dv = lambda t: hypot(self.ddx(t), self.ddy(t))
        self.do = lambda t: arctan2(self.ddy(t), self.ddx(t))
        #steering (change of orientation)
        self.s = lambda t: self.o(t+.5)-self.o(t-.5)
        #acceleration
        self.a = lambda t: self.v(t+.5)-self.v(t-.5)


if __name__ == '__main__':
#examples:
    total_time = lambda trip: trip.n
    total_distance = lambda trip: trip.distance_driven
    straight_distance = lambda trip: trip.distance_straight
    straightness = lambda trip: trip.straightness
    sum_turnspeeds = lambda trip: sum(trip.s(trip.t)/trip.v(trip.t))**2
    acceleration_to_dist = lambda trip: sum(trip.a(trip.t)**2)
    
    total_standstill_time = lambda trip: np.count_nonzero(trip.v(trip.t) < 0.1)
    
    trippath = 'D:\\Documents\\Data\\MLiP\\drivers\\1\\1.csv'    
    tripp = trip(trippath)
    
    print(tripp.v(tripp.t))
    print(total_standstill_time(tripp))