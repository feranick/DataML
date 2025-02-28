#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* MasterDataMaker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: v2025.02.27.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py, pickle
from random import uniform
from bisect import bisect_left
from libDataML import *

#************************************
# Parameters definition
#************************************
class dP:
    
    saveAsTxt = True

    numHeadColumns = 2
    numHeadRows = 0
    
    fullDataset = True
    minCCol = 1
    maxCCol = 42
    #charCCols = [8,10,12,13]
    charCCols = [21,23,25,34]
    #charCCols = [1,2,3]
    predRCol = [42]
    
    purgeUndefRows = False
    
    validFile = True
    createRandomValidSet =  False
    percentValid = 0.05
    validRows = [1,2,3]
    
    precData = 3
    saveNormalized = False
    normalizeLabel = False
    useCustomRound = True
    YnormTo = 1
    stepNormLabel = 0.001
    
    randomize = False
    fullRandomMatrix= True
    numRandomAdds = 50
    randomCols = [3]
    minPercVariation = 0.05
    randomizeLabel = False

    useGeneralNormLabel = False
    minGeneralLabel = 10
    maxGeneralLabel = 60
    
    valueForNan = -1
    
    # Do not change
    
    def rescaleList(list, value):
        list = [x + value for x in list]
        return list
    
    if fullDataset:
        minCCol = minCCol + numHeadColumns-1
        maxCCol = maxCCol + numHeadColumns-1
    else:
        rescaleList(charCCols, numHeadColumns - 1)
        
    predRCol = rescaleList(predRCol, numHeadColumns - 1)

    numLabels = len(predRCol)
    validRows = [x-1 for x in validRows]

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 DataMaker.py <paramFile> <pred column - optional>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    if len(sys.argv) >= 3:
        predRCol = [int(sys.argv[2])]
    else:
        predRCol = dP.predRCol
    
    if dP.fullDataset:
        datasetLabel = '_fullDataSet'
    else:
        datasetLabel = '_partialDataSet'
    
    rootFile = os.path.splitext(sys.argv[1])[0] + datasetLabel
    
    if dP.purgeUndefRows:
        rootFile += '_purged'
        
    rootFile += '_p' + str(predRCol[0])
    learnFile = rootFile + '_train'
    
    #try:
    P,V = readParamFile(sys.argv[1], predRCol, rootFile)
    #except:
    #    print("\033[1m" + " param file not found \n" + "\033[0m")
    #    return
    
    #************************************
    # Creating training set
    #************************************
    if dP.randomize:
        print(" Creating randomized training set using",dP.minPercVariation*100, "% as max variation on parameters\n")
        Pr = randomize(P)
        
        if dP.fullRandomMatrix:
            randTag = '_fullrand'
        else:
            randTag = '_partrand'
       
        learnRandomFile = learnFile + randTag + str(dP.numRandomAdds) + '_pcVar' + str(int(dP.minPercVariation*100))
        
        if dP.randomizeLabel:
            learnRandomFile += '_rLab'
        if dP.saveNormalized or dP.normalizeLabel:
            norm = Normalizer(Pr, dP)
            norm.save(rootFile +"_norm.pkl")
            saveLearnFile(norm.transform_matrix(Pr), learnRandomFile+'_norm', True)
        else:
            saveLearnFile(Pr, learnRandomFile, False)
    else:
        if dP.saveNormalized or dP.normalizeLabel:
            norm = Normalizer(P, dP)
            norm.save(rootFile+ "_norm.pkl")
            saveLearnFile(norm.transform_matrix(P), learnFile +'_norm', True)
        else:
            saveLearnFile(P, learnFile, False)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, predRCol, rootFile):
    if dP.fullDataset:
        usecols = range(dP.minCCol,dP.maxCCol)
    else:
        usecols = dP.charCCols
    
    with open(paramFile, 'r') as f:
        P2 = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows).to_numpy()
    
    M = np.hstack((P2[:,predRCol],P2[:,usecols]))

    if dP.purgeUndefRows:
        M = purgeRows(M)
    
    P = np.vstack([list(range(0,M.shape[1])),M])
    V = np.array([list(range(0,M.shape[1]))])

    #***************************************
    # Handle Validation File
    #***************************************
    if dP.validFile:
        validFile = rootFile + '_test'
        
        if dP.createRandomValidSet:
            P, V = formatSubset(P, dP.percentValid)
        else:
            if dP.validRows != 0:
                P = np.vstack([list(range(0,M.shape[1])),np.delete(M,dP.validRows,0)])
                V = np.vstack([list(range(0,M.shape[1])),M[dP.validRows,:]])
                
        if dP.saveNormalized or dP.normalizeLabel:
            saveLearnFile(norm.transform_matrix(V), validFile +'_norm', True)
        else:
            saveLearnFile(V, validFile, False)
    return P,V
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile, saveNormFlag):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'ab') as f:
                 np.savetxt(f, M, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        if saveNormFlag == False:
            print(" Saving new training file (txt) in:", learnFile+"\n")
        else:
            print(" Saving new normalized training file (txt) in:", learnFile+"\n")
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M.astype(np.float64))

#***************************************
# Randomize initial set
#***************************************
def randomize(P):
    Pr = np.copy(P)

    if not dP.fullRandomMatrix:
        cols = dP.randomCols
    else:
        cols = P[0,1:].astype(int)
        
    for j in range (0,dP.numRandomAdds):
        rand = randomMatrix(P, cols)
        temp = np.multiply(P[1:],rand)
        Pr = np.vstack([Pr, temp])
    return Pr

def randomMatrix(P, cols):
    rand = np.ones(P[1:].shape)
    rand[:,cols] = np.random.uniform(1-dP.minPercVariation,1,rand[:,cols].shape)
    return rand
    
#************************************
# Create validation subset
#************************************
def formatSubset(A, percent):
    from sklearn.model_selection import train_test_split
    
    numValid = round(A[1:,:].shape[0]*percent)
    numTrain = round(A[1:,:].shape[0] - numValid)
    print(" Creating a training set with:", str(numTrain), "datapoints")
    print(" Creating a validation set with:", str(numValid), "datapoints\n")
    
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A[1:,1:], A[1:,0], test_size=percent, random_state=42)
    Atrain = np.vstack((A[0,:],np.hstack((Cl_train.reshape(1,-1).T, A_train))))
    Atest = np.vstack((A[0,:],np.hstack((Cl_cv.reshape(1,-1).T, A_cv))))
    
    return Atrain, Atest
'''
def formatSubset2(A, Cl, percent):
    list = np.random.choice(range(len(Cl)), int(np.rint(percent*len(Cl))), replace=False)
    A_train = np.delete(A,list,0)
    Cl_train = np.delete(Cl,list)
    A_cv = A[list]
    Cl_cv = Cl[list]
    return A_train, Cl_train, A_cv, Cl_cv
'''

#************************************
# Purge rows with undefined Cl value
#************************************
def purgeRows(M):
    condition = M[:,0] == 0
    M2 = M[~condition]
    print(" Shape original dataset:", M.shape)
    print(" Purged from undefined values. \n Shape new dataset:",M2.shape,"\n")
    return M2

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
