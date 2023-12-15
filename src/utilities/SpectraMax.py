#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* SpectraMax
* Finds Max for spectral data
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
from scipy import stats
import sys, os.path, h5py
from random import uniform
from bisect import bisect_left
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
    
    #trainCol = [3,3553]   # Raw data
    trainCol = [3,80000]   # Raw data
    predCol = [1,3]       # Raw data
    
    #trainCol = [28,47]     # ML3
    #predCol = [5,7]        # ML-3
    
    #trainCol = [7,54]
    #predCol = [1,7]
    #trainCol = [61,106]
    #predCol = [28,47]
    #predCol = [61,106]

    valueForNan = -1
    validRows = [40,41,42,43]
    
    corrMin = .8
    corrMax = 1
    #corrMin = -1
    #corrMax = -.7

    heatMapsCorr = False            # True: use for Master data
    plotGraphs = False
    plotGraphsThreshold = False
    plotValidData = True
    plotLinRegression = True
    graphX = [8,10,12,13,14]
    graphY = [62,69,78,79,80,81]
    
    plotCorr = True                # True: use for raw data (spectra, etc)
    stepXticksPlot = 1500
    
    polyDegree = 1

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 SpectraMax <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dfPMax = readParamFile(sys.argv[1], dP.trainCol)
    #V,headV,_ = readParamFile(sys.argv[1], dP.predCol)
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    maxFile = rootFile + '_max.csv'
    dfPMax.to_csv(maxFile, index=False, header=True)
    print("\n Max for spectral data saved in in:",maxFile,"\n")

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, lims):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
            if lims[1]>len(dfP.columns):
                lims[1] = len(dfP.columns)
                print(" Warning: Column range is larger than actual number of column. Using full dataset")
            
            P = dfP.iloc[:,range(lims[0],lims[1])].to_numpy()
            P[np.isnan(P)] = dP.valueForNan

        with open(paramFile, 'r') as f:
            headP = np.genfromtxt(f, unpack = False, usecols=range(lims[0],lims[1]),
                delimiter = ',', skip_header=dP.skipHeadRows, skip_footer=P.shape[0], dtype=np.str)

    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return

    dfPMax = pd.DataFrame()
    dfPMax[""] = dfP.iloc[:,0]
    dfPMax["Max-X"] = headP[np.argmax(P, axis=1)]
    dfPMax["Max-Y"] = np.amax(P, axis=1)
    
    print(dfPMax)
    
    return dfPMax

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
