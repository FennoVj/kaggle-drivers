import numpy as np

from pyprocessor import trip, tripc, full_data
from numpy import array


total_time = lambda trip: trip.n
total_distance = lambda trip: trip.D
straight_distance = lambda trip: trip.distance_straight
straightness = lambda trip: trip.straightness
sum_turnspeeds = lambda trip: sum(trip.s(trip.t)/trip.v(trip.t))**2
acceleration_to_dist = lambda trip: sum(trip.a(trip.t)**2)
ms = lambda trip: [trip.distances[i] for i in range(7)]

def diff(i):
    def v(x):
        V = np.zeros(len(x))
        V[:-1] = x[1:,i]-x[:-1,i]
        V[-1] = V[-2]
        return V
    return v

#dt, dd, dx, dy = diff(0), diff(1), diff(2), diff(3)

def speed(x):
    return dd(x)/dt(x)

def direction(x):
    return np.arctan2(dy(x), dx(x))

features = [total_time, total_distance, straight_distance, straightness, sum_turnspeeds, acceleration_to_dist]
#from numpy import *
#n = 20
#features = (n, [real, imag, abs, angle], speed, direction, diff(6), diff(7))


def main(driver_id, out='output'):
    from os import listdir, path

    folder = '%s/%s'%(full_data, driver_id)
    tripfiles = ['%s/%s'% (folder, f) for f in listdir(folder)]
    #feats = np.empty((len(tripfiles), len(features)+7))

    #The following 5 lines are added by fenno to fix a bug in the code
    driverfolder = path.dirname(tripfiles[0]) + '/'
    basenames = [path.basename(f) for f in tripfiles]
    basenames = [int(f[:-4]) for f in basenames]
    basenames =  [str(f) + '.csv' for f in sorted(basenames)]
    tripfiles = [driverfolder + b for b in basenames]

#    feats = array([tripc(file).features(n, [real, imag, abs, angle], speed, direction, diff(6), diff(7)).ravel() for file in tripfiles])
    feats =  array([[f(trip(t)) for f in features] for t in tripfiles]).T
    #for i, file in enumerate(tripfiles):
    #    t = trip(file)
     #   t = tripc(file)
    #    fs = [f(t) for f in features] + [f for f in t.distances]
    #    print(len(fs))
    #    feats[i] = np.array(fs)
        #feats[i] = np.array([f(t) for f in features] + list(ms(t)))
    write(driver_id, feats, out=out)
    print(driver_id)


def write(driver, feats, out='output'):
    from csv import writer
    with open('./%s/%s.csv'% (out, driver),'w') as outfile:
        writer(outfile).writerows(feats)

if __name__=='__main__':

    from sys import argv

    folder = argv[1]
    if len(argv) > 2:
        out = argv[2]
        main(folder, out)
    main(folder)

