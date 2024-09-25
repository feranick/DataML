#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**************************************************
* CreateMix4CorrAnalysis
* Create mixture of data for Correlation analysis
* version: v2024.9.25.3
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
**************************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
from scipy import stats
import sys, os.path, h5py
from random import uniform
from bisect import bisect_left
from itertools import permutations
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
    
    ### Master Data Handling
    specifyColumns = False
    trainCol = [1, 40]       # IGC (column range)
    predCol = [41, 48]       # IGC (column range)
    #trainCol = [1, 48]       # IGC (column range)
    #predCol = [1, 48]       # IGC (column range)
    #trainCol = [1,2,3]       # IGC (column range)
    #predCol = [4,5,6]       # IGC (column range)
    
    numLastPredCol = 9
    
    separateValidFile = False
    validRows = [103,104,105,106,107]   # ORNL

    valueForNan = -1
    removeNaNfromCorr = True

    if specifyColumns == False:
        trainCol = [item for item in range(trainCol[0], trainCol[1]+1)]
        predCol = [item for item in range(predCol[0], predCol[1]+1)]

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 MixCorrAnalysis <paramFile>')
        print(' Usage:\n  python3 MixCorrAnalysis <paramFile> <validFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    newFile = rootFile+"_mix.csv"
    dfP = readParamFile(sys.argv[1])
    if dP.separateValidFile:
        dfV = readParamFile(sys.argv[2])
        dfP = dfP.append(dfV,ignore_index=True)
        dP.validRows = dfP.index.tolist()[-len(dfV.index.tolist()):]
    
    
    P,headP = processParamFile(dfP, dP.trainCol)
    V,headV = processParamFile(dfP, dP.predCol)
    
    permList = getPermutations(dP.trainCol)

    #for p in permList:
    #    addSumColumn(dfP, p)
    
    dfP = addSumColumn(dfP, permList)
    
    dfP.to_csv(newFile, index=False, header=True)
    print(" New parameter file saved in:",newFile,"\n")
    
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
        print(dfP)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return dfP

def processParamFile(dfP, lims):
    if lims[1]>len(dfP.columns):
        lims[1] = len(dfP.columns)
        print(" Warning: Column range is larger than actual number of columns. Using full dataset")
    #P = dfP.iloc[:,lims].astype(float).to_numpy()
    P = dfP.iloc[:,lims].to_numpy()
    headP = dfP.columns[lims].values
    P[np.isnan(P)] = dP.valueForNan
    return P, headP
    
#************************************************************************
# Create permutations and add to dataframe as extra columns
#************************************************************************

def addSumColumn(dfP, perm):
    P, headP = [], []
    for p in perm:
        P.append(dfP.iloc[:,p[0]].to_numpy() + dfP.iloc[:,p[1]].to_numpy())
        headP.append(dfP.columns[p[0]] + "+" + dfP.columns[p[1]])

    df_temp = pd.DataFrame(data=np.array(P).T, columns=headP)
    pos = len(dfP.columns)-dP.numLastPredCol
    df = pd.concat([dfP.iloc[:,:pos],df_temp, dfP.iloc[:, pos:]], axis=1)
    print(df)
    print("\n Added", df_temp.shape[1],"columns to the original", pos)
    print(" Total number of training columns:",df.shape[1]-dP.numLastPredCol,"\n Prediction columns:",dP.numLastPredCol,"\n")
    return df

def getPermutations(l):
    from itertools import permutations
    permList = []
    for p in permutations(l, 2):
        if p[0] <= p[-1]:
            permList.append(p)
    print("\n Created",len(permList), "permutations.\n")
    return permList

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

