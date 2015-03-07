
import numpy as np

from pyprocessor import trip, full_data


total_time = lambda trip: trip.n
total_distance = lambda trip: trip.distance_driven
straight_distance = lambda trip: trip.distance_straight
straightness = lambda trip: trip.straightness
sum_turnspeeds = lambda trip: sum(trip.s(trip.t)/trip.v(trip.t))**2
acceleration_to_dist = lambda trip: sum(trip.a(trip.t)**2)



features = [total_time, total_distance, straight_distance, straightness, sum_turnspeeds, acceleration_to_dist]

def main(driver_id):
    from os import listdir, path

    folder = '%s/%s'%(full_data, driver_id)
    tripfiles = ['%s/%s'% (folder, f) for f in listdir(folder)]

    feats = np.empty((len(tripfiles), len(features)))

    #The following 5 lines are added by fenno to fix a bug in the code
    driverfolder = tripfiles[0][:-5] #kinda dirty hack, only works if 1.csv is first file in folder
    basenames = [path.basename(f) for f in tripfiles]
    basenames = [int(f[:-4]) for f in basenames]
    basenames =  [str(f) + '.csv' for f in sorted(basenames)]
    tripfiles = [driverfolder + b for b in basenames]


    for i, file in enumerate(tripfiles):
        t = trip(file)
        feats[i] = np.array([f(t) for f in features])
    write(driver_id, feats)
    print(driver_id)


def write(driver, feats):
    from csv import writer
    with open('./output/%s.csv'% driver,'w') as outfile:
        writer(outfile).writerows(feats)

if __name__=='__main__':

    from sys import argv

    folder = argv[1]
    main(folder)
