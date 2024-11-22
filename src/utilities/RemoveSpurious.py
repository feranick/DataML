#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
************************************************
* RemoveSpurious
* Remove spurious data from training set 
* that have values below the minimum allowed
* version: v2024.11.22.2
* By: Nicola Ferralis <feranick@hotmail.com>
************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 RemoveSpurious.py <learnDataOrig> <learnDataNew> ')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, Aorig, Morig = readLearnFile(sys.argv[1])
    En, A, M = readLearnFile(sys.argv[2])
    newFile = os.path.splitext(sys.argv[2])[0] + '_removedSpurious'
    
    newTrain = removeSpurious(En, A, Aorig)
    saveLearnFile(dP, newTrain, newFile, "")

#******************************************************
# Create new Training data by adding a percentage of the max
# for that feature
#******************************************************
def getAmin(A):
    A_min = []
    for i in range(A.shape[1]):
        A_min_single = min(x for x in A[:,i] if x != 0)
        A_min = np.hstack([A_min,A_min_single])
    A_min = np.asarray(A_min)
    return A_min
        
def removeSpurious(En, A, Aorig):
    A_min = getAmin(Aorig)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < A_min[j]:
                A[i,j] = 0
    newTrain = np.vstack([En, A])
    return newTrain

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(dP, M, learnFile, tag):
    learnFileRoot = os.path.splitext(learnFile)[0]
    if dP.saveAsTxt == True:
        learnFileRoot += tag + '.txt'
        print(" Saving new training file (txt) in:", learnFileRoot+"\n")
        with open(learnFileRoot, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFileRoot += tag + '.h5'
        print(" Saving new training file (hdf5) in: "+learnFileRoot+"\n")
        with h5py.File(learnFileRoot, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
