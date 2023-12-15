#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* TranposeData
* Transpose training data set
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
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

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 TransposeData <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]+'_T.csv'
    dfP = readParamFile(sys.argv[1])
    dfP.to_csv(TFile, index=True)

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
    print(dfP.T)
    return dfP.T

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
