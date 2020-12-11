#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Add data with random noise on all colums or
* on selected ones.
* Noise is a random percentage multiplier

* version: 20201211a
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    addToFlatland = False
    Ynorm = True
    YnormTo = 1
    fullRandomMatrix= True
    minPercVariation = 0.1
    randomCols = [2]
    
    randomCols[0]-=1
    
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 4:
        print(' Usage:\n  python3 AddNoisyData.py <learnData> <#additions>')
        print('  Data is by default normalized to 1\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    newFile = os.path.splitext(sys.argv[1])[0] + '_n' + sys.argv[2] + '_pcVar' + str(int(dP.minPercVariation*100))
    learnFileExt = os.path.splitext(sys.argv[1])[1]
        
    print(" Creating randomized training set using",dP.minPercVariation*100, "% as max variation on parameters\n")

    En, M = readLearnFile(sys.argv[1])
    
    if dP.Ynorm ==True:
        print(" Normalizing Learning Spectra to:",dP.YnormTo)
        M = normalizeSpectra(M)
        newFile += '_norm1'

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    for j in range(int(sys.argv[2])):
        newTrain = np.vstack((newTrain, randomize(M)))

    if dP.Ynorm ==True:
        print(" Normalizing Learning + Noisy Spectra to:",dP.YnormTo,"\n")
        newTrain = normalizeSpectra(newTrain)

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
# Introduce Noise in Data
#************************************
def randomize(M):
    rand = randomMatrix(M[:,1:])
    M[:,1:] = np.multiply(M[:,1:],rand)
    return M

def randomMatrix(P):
    random = np.random.uniform(1-dP.minPercVariation,1,P.shape[1])
    if not dP.fullRandomMatrix:
        rand = np.ones(P.shape[1])
        rand[dP.randomCols] = random[dP.randomCols]
    else:
        rand = random
    return rand

#************************************
# Normalize
#************************************
def normalizeSpectra(M):
    for i in range(1,M.shape[0]):
        if(np.amin(M[i]) <= 0):
            M[i,1:] = M[i,1:] - np.amin(M[i,1:]) + 1e-8
        #M[i,1:] = np.multiply(M[i,1:], dP.YnormTo/max(M[i][1:]))
    M[1:,1:] = np.multiply(M[1:,1:], np.array([float(dP.YnormTo)/np.amax(M[1:,1:], axis = 1)]).T)
    return M

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
