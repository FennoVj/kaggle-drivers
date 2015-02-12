# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:17:45 2015

@author: Fenno Vermeij
"""

import numpy as np

"""
Appends the probabilities for a given driver to a file
This will append "\n" first, then the actual probabilities, and then "x_" at the end
Therefore, calling appendprobabilities multiple times makes a file that is not ready for submission yet
To create a file ready for submission, call 'createsubmissionfile' after appending all the probabilities

filename: the filename of where to append the probababilities of this driver
drivernr: the number of the driver
probs: the 200-by-2 matrix with the first column containing integers (the trip nrs), 
  and the second column containing the probability that that trip belongs to the driver
fmtstring: how to format the probability. By default, this argument does not need to filled in if 
  the probability is either 0 or 1. If there are also decimals, the format string needs to be '%0.xf', 
  where f is the number of significant digits the probability needs to have
"""
def appendProbabilities(filename, drivernr, probs, fmtstring = '%0.0f'):
    probs = np.sort(probs,axis=0)
    with open(filename, 'a') as f_handle:
        f_handle.write("\n" + str(drivernr) + "_")
        np.savetxt(f_handle, probs, header = "", footer = "", delimiter= ",", fmt=['%0.0f',fmtstring], newline = "\n" + str(drivernr) + "_", comments = "")


"""
Takes a file that is created by multiple calls of appendProbabilities, and creates a submissionfile

infilename: the filename that contains the probabilities from the 'appendProbabilities' method
outfilename: the file that needs to contain the submission file. Can be the same as the infilename,
  but in this case, all the existing content in this file will be deleted and overwritten with 
  the actual submission file
"""
def createSubmissionfile(infilename, outfilename):
    with open(infilename, 'r') as f_handle:
        data = f_handle.read()

    lines = data.split('\n')
    for i in range(len(lines)):
        if lines[i].endswith('_'):
            lines[i] = ""
        else:
            lines[i] = lines[i] + "\n"

    with open(outfilename, 'w') as f_handle:
        f_handle.write("driver_trip,prob\n")
        f_handle.write("".join(lines[1:]))    


"""
As an example, make a submission file of three drivers, all with the same probabilities
The file "foo2" contains the preliminary data
The file "foo3.csv" contains the submission file for this example with only 3 drivers and 3 trips each
"""
def submissionfileExample():
    filename = "foo2"
    open(filename, 'w').close() #empty the file
    outfilename = "foo3.csv"
    probs = np.asarray([[3,0],[4,1],[2,1]])
    appendProbabilities(filename, 1, probs)
    appendProbabilities(filename, 2, probs)
    appendProbabilities(filename, 3, probs)
    createSubmissionfile(filename, outfilename)
        