#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* SplitXRange
* Split X Range in chuncks
*
* version: 20190910a
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

    rangeMin = 600
    rangeMax = 700

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 SplitXRange <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]
    TFile += '_Interp'+str(dP.rangeMin)+'-'+str(dP.rangeMax)+'.csv'
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
    sR = int(dP.skipNumRows)
    print(dfP)

    dfP = dfP[0:sR].append(dfP[sR:][(dfP.index.values[sR:].astype(float) >= int(dP.rangeMin)) & (dfP.index.values[sR:].astype(float)<int(dP.rangeMax))])
        
    print(dfP)
    return dfP

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
