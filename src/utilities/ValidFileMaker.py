#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* ValidFileMaker
* Make Single Validation File
* version: v2024.10.04.1
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    numHeadRows = 0
    row = 1
    
    fullDataset = False
    minCCol = 1
    maxCCol = 10
    charCCols = [14,21,23,29,32,34,35,36,37,38,39,40]
    
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
    
    if len(sys.argv) >= 3:
        row = int(sys.argv[2])
    else:
        row = dP.row
    try:
        M, name = readParamFile(sys.argv[1], sys.argv[2])
    except:
        print("\033[1m" + " Param file not found \n" + "\033[0m")
        return
        
    rootFile = os.path.splitext(sys.argv[1])[0]
    validFile = rootFile + '_val-row_' + str(row) + '_name_' +str(name)
    saveLearnFile(M, row, name, validFile)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, row):
    if dP.fullDataset:
        usecols = range(dP.minCCol,dP.maxCCol)
    else:
        usecols = dP.charCCols

    with open(paramFile, 'r') as f:
       if os.path.splitext(paramFile)[1] == ".csv":
            import pandas as pd
            P2 = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows).to_numpy()
       if os.path.splitext(paramFile)[1] == ".txt":
            P2 = np.loadtxt(f, unpack =True)
            
    M = P2[:,usecols]
    return M, P2[int(row),0]
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, row, name, learnFile):
    print(" Using row:",row,"- Name:",name,"\n")
    En = np.arange(0,M[row,:].size,1)
    M1 = np.array([En,M[row,:]]).T
    print(M1)

    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'ab') as f:
                 np.savetxt(f, M1, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        print("\n Saving new validation file (txt) in:\n ", learnFile+"\n")
    else:
        learnFile += '.h5'
        print("n Saving new validation file (hdf5) in: \n "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M1)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
