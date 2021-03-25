#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* ValidFileMaker
* Make Single Validation File
*
* version: 20210325a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
**********************************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py
from random import uniform
from bisect import bisect_left
from libDataML import *

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    numHeadRows = 0
    row = 1
    
    fullDataset = True
    minCCol = 1
    maxCCol = 10
    #charCCols = [8,10,12,13]
    charCCols = [6,8,13]
    
    precData = 3
    valueForNan = -1

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 ValidFileMaker.py <paramFile> <row>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    if len(sys.argv) >= 4:
        row = [int(sys.argv[2])]
    else:
        row = dP.row
    try:
        M = readParamFile(sys.argv[1])
    except:
        print("\033[1m" + " Param file not found \n" + "\033[0m")
        return
        
    rootFile = os.path.splitext(sys.argv[1])[0]
    validFile = rootFile + '_valid-row_' + str(row)
    saveLearnFile(M, row, validFile)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile):
    if dP.fullDataset:
        usecols = range(dP.minCCol,dP.maxCCol)
    else:
        usecols = dP.charCCols
    
    with open(paramFile, 'r') as f:
        P2 = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows).to_numpy()
    
    #M = np.hstack((P2[:,predRCol],P2[:,usecols]))
    M = P2[:,usecols]
    #----------------------------------
    return M
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, row, learnFile):
    print(" Using row:",row,"\n")
    En = np.arange(0,M[row,:].size,1)
    M1 = np.array([En,M[row,:]]).T
    print(M1)

    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'ab') as f:
                 np.savetxt(f, M1, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        print("\n Saving new validation file (txt) in:", learnFile+"\n")
    else:
        learnFile += '.h5'
        print("n Saving new validation file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M1)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
