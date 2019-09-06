#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* NormColumns
* Normalize columns for master data
*
* version: 20190506a
*
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
*
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
    valueForNan = -1
    skipNumRows = 2

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 NormColumns <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]
    TFile += '_Norm.csv'
    dfP = readParamFile(sys.argv[1])

    dfPint = normalize(dfP)

    dfPint.to_csv(TFile, index=True)

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
    for i in range(len(dfP.columns)):
        print("Column:",i," - ",dfP.columns[i]," - Max:",dfP.iloc[sR:,i].max(),
            " - Min:",dfP.iloc[sR:,i].min())
        dfPnorm.iloc[sR:,i] = (dfP.iloc[sR:,i]-dfP.iloc[sR:,i].min())/dfP.iloc[sR:,i].max()
    
    print("\n",dfPnorm)
    return dfPnorm

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
