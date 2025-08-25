#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CheckDataSet
* version: 2025.08.25.1
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, configparser, ast

#***************************************************
# This is needed for installation through pip
#***************************************************
def CheckDataSet():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
        #Kyo Data
        #self.skipHeadRows = 5
        #self.skipHeadColumns = 7
        #self.rowParameters = 5
        
        #IGC Poly data
        self.skipHeadRows = 3
        self.skipHeadColumns = 1
        self.rowParameters = 3
        
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 CheckDataSet <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dP = Conf()
        
    dfP, dfL = readParamFile(sys.argv[1], dP)
    if dfP is None:
        print("Exiting due to error reading parameter file.")
        return 0
        
    checkBadTypesDataset(dfP, dfL)
        
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, dP):
    try:
        with open(paramFile, 'r') as f:
            dfPorig = pd.read_csv(f, delimiter = ",", header=None)
            print(dfPorig)
            dfL=dfPorig.iloc[dP.rowParameters-1].tolist()[1:]
            print(dfL)
            
            dfP = dfPorig.iloc[dP.skipHeadRows:].iloc[:, (dP.skipHeadColumns):]
            print(dfP)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return 0
    return dfP, dfL
    
#*************************************************************
# Check and identify where are badly typed entries in dataset
#*************************************************************
def checkBadTypesDataset(dfP, dfL):
    non_numeric_cols = dfP.select_dtypes(exclude=np.number).columns
    numDetected = 0
    
    for i in range(1,len(non_numeric_cols),1):
        non_numeric_mask = pd.to_numeric(dfP[non_numeric_cols[i]], errors='coerce').isnull()
        non_numeric_members = dfP.loc[non_numeric_mask, non_numeric_cols[i]]
        if len(non_numeric_members) > 0:
            print("\n---------------------------------")
            print(f"Column: {dfL[i]}")
            print("---------------------------------")
            print("Row    Value")
            print(non_numeric_members)
            numDetected+=1

    if numDetected == 0:
        print(" No non-numerical values found. All good.\n")
    else:
        print("\n")

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
