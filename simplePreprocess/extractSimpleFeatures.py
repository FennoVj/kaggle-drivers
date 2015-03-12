import numpy as np

from simplePyprocessor import trip
from simpleFeatures import features
from driverFeatures import driverfeatures, makeDriverFeature
from os import listdir, path

full_data = '../data/drivers'

def getdrivernrs(tdatpath):
    files = listdir(tdatpath)
    return np.array(sorted([int(f) for f in files]))

def main(driver_id, numT = 200):


    folder = '%s/%s'%(full_data, driver_id)
    tripfiles = ['%s/%s'% (folder, f) for f in listdir(folder)]
    drivernrs = getdrivernrs(full_data)
    numR = len(tripfiles)
    simF = len(features) #simple features
    advF = len(driverfeatures)

    feats = np.empty((numR + numT, simF + advF))

    #The following 5 lines are added by fenno to fix a bug in the code
    driverfolder = path.dirname(tripfiles[0]) + '/'
    basenames = [path.basename(f) for f in tripfiles]
    basenames = [int(f[:-4]) for f in basenames]
    basenames =  [str(f) + '.csv' for f in sorted(basenames)]
    realtripfiles = [driverfolder + b for b in basenames]
    
    driverprobs = np.ones(len(drivernrs))
    driverprobs[drivernrs == driver_id] = 0
    driverprobs = driverprobs / float(len(drivernrs) - 1)
    randD = np.random.choice(drivernrs, size = numT ,replace=False, p = driverprobs)
    randT = np.random.randint(1 ,numT+1, size = numT) 
    fakefolder = ['%s/%s'%(full_data, did) for did in randD]
    faketripfiles = ['%s/%s.csv'%(fakefolder[i], randT[i]) for i in range(numT)]
        
    realtrips = []
    for i, file in enumerate(realtripfiles):
        t = trip(file)
        realtrips.append(t)
        feats[i,:simF] = np.array([f(t) for f in features])
        
    faketrips = []
    for i, file in enumerate(faketripfiles):
        t = trip(file)
        faketrips.append(t)
        feats[i+numR,:simF] = np.array([f(t) for f in features])
        
    for i, f in enumerate(driverfeatures):
        feats[:,simF + i] = makeDriverFeature(realtrips, faketrips, f)
    
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
#cd output
#zip -r ../features.zip *
#cd ..
