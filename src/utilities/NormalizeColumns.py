#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* NormalizeColumns
* Normalize columns for master data
* version: 20201217a
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, math

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 1
    valueForNan = -1
    skipNumRows = 0
    normAllCols = True
    numColsNorm = 60
    saveAsLog = False
    prec=0.001

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 NormalizeColumns <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]
    if dP.saveAsLog:
        TFile += '_LogNorm.csv'
    else:
        TFile += '_Norm.csv'
    
    dfP = readParamFile(sys.argv[1])
    dfPint = normalize(dfP)
    dfPint.to_csv(TFile, index=True)
    print("\n Saving normalized file in:",TFile,"\n")

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",",
                skiprows=dP.skipHeadRows, index_col=[0], na_values=dP.valueForNan)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    print(dfP)
    return dfP

#************************************
# Interpolate X Range
#************************************
def normalize(dfP):
    dfPnorm = dfP.copy()
    sR = int(dP.skipNumRows)
    if dP.normAllCols:
        normCols= range(len(dfP.columns))
    else:
        normCols = range(0,dP.numColsNorm)
    
    print(normCols)
        
    for i in normCols:
        data = dfP.iloc[sR:,i]
        
        print("Column:",i," - ",dfP.columns[i]," - Max:",data.max(),
            " - Min:",data.min())
        
        if dP.saveAsLog:
            if data.min() < 0:
                data = data-data.min()+dP.prec
            data = np.log10(data)
        
        dfPnorm.iloc[sR:,i] = (data-data.min())/data.max()
    print("\n",dfPnorm)
    return dfPnorm

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())

