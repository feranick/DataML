#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Stack shuffled learning data
* version: v2024.12.16.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
from libDataML import *

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
        print(' Usage:\n  python3 ShuffleData.py <learnData> <#additions>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    learnFileExt = os.path.splitext(sys.argv[1])[1]
        
    En, A = readLearnFile(sys.argv[1])
    
    print(A.shape[0])
    print(sys.argv[2])
    numAdditions = str(A.shape[0]*(int(sys.argv[2])+1))
    
    print(" Adding "+sys.argv[2]+" new datasets (total datapoints: "+numAdditions+")\n")
    newFile = os.path.splitext(sys.argv[1])[0] + '_nShuffle-' + numAdditions

    
    A_tmp = addShuffled(A, int(sys.argv[2]))
    
    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, A_tmp))
    else:
        newTrain = A_tmp

    saveLearnFile(newTrain, newFile)

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

    En = M[0,1:]
    A = M[1:,:]
    #Cl = M[1:,0]
    return En, A


#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
# Create shuffled data
#************************************
def addShuffled(P, numRandomAdds):
    Pr = np.copy(P)
    for i in range(numRandomAdds):
        np.random.shuffle(Pr)
        P = np.vstack((P, Pr))
        
    return P

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
