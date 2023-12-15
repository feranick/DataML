#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* SplitXRange
* Split X Range in chuncks
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
    skipNumRows = 2

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 SplitXRange <paramFile> <rangeMin> <rangeMax>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]
    TFile += '_Interp'+str(sys.argv[2])+'-'+str(sys.argv[3])+'.csv'
    print(" Saving data with restricted X range into:",TFile,"\n")
    dfP = readParamFile(sys.argv[1], sys.argv[2], sys.argv[3])
    dfP.to_csv(TFile, index=True)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, rMin, rMax):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",",
                skiprows=dP.skipHeadRows, index_col=[0], na_values=dP.valueForNan)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    sR = int(dP.skipNumRows)
    print(dfP)

    dfP = dfP[0:sR].append(dfP[sR:][(dfP.index.values[sR:].astype(float) >= int(rMin)) & (dfP.index.values[sR:].astype(float)<int(rMax))])
        
    print(dfP)
    return dfP

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
