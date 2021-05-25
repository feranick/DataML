#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Add Noisy Data to CSV
* version: 20210524b
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
    addFixOffset = True
    customOffset = True
    cOffset = [10,10,10,10,10,10,10,10,10,30,0.5,30,0]
    #cOffset = [0,0,0,0,0,0,0,0,0,50,0,0,0]
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 4 and dP.customOffset == False:
        print(' Usage:\n  python3 AddNoisyDataCSV <paramFile> <#additions> <offset>')
        print('  Offset: multiplier in units of percent (i.e. 1 is 1%)')
        print('  Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dfP = readParamFile(sys.argv[1])
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    noisyFile = rootFile + '_noisy-'
    
    if dP.customOffset == True:
        noisyFile += 'cust.csv'
        offs = dP.cOffset
    else:
        noisyFile += 'opc'+sys.argv[3]+'.csv'
        offs = int(sys.argv[3])

    dfP_noise = addNoise(dfP, int(sys.argv[2]), offs)
    dfP_noise.to_csv(noisyFile, index=False, header=True)
    
    print(sys.argv[2],"iterations (offset:",offs,") \nSaved in:",noisyFile,"\n")
    
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return dfP

#************************************
''' Introduce Noise in Data '''
#************************************
def addNoise(dfP, num, offset):
    dfP_temp = dfP.copy()
    dfP_noise = dfP.copy()
    for i in range(1, num):
        factor = offset*np.random.uniform(-0.01,0.01,(dfP_temp.iloc[:,1:].shape))
        dfP_temp.iloc[:,1:] = dfP.iloc[:,1:].mul(1+factor)
        dfP_noise = dfP_noise.append(dfP_temp, ignore_index=True)
        
    #print(dfP_noise[dfP_noise["Specimen"] == "2194"])
    return dfP_noise

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
