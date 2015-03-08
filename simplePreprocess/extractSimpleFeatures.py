import numpy as np

from simplePyprocessor import trip
from simpleFeatures import features

full_data = '../data/drivers'

def main(driver_id):
    from os import listdir, path

    folder = '%s/%s'%(full_data, driver_id)
    tripfiles = ['%s/%s'% (folder, f) for f in listdir(folder)]

    feats = np.empty((len(tripfiles), len(features)))

    #The following 5 lines are added by fenno to fix a bug in the code
    driverfolder = path.dirname(tripfiles[0]) + '/'
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

        #sample usage:   
        #parallel -j 24 python ./extractSimpleFeatures.py -- `ls ../data/drivers`
        #rm features.zip
        #cd fennoOutput
        #zip -r ../features.zip *
