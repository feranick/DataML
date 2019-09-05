#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* InterpXRange
* Interpolate X Range for master data
*
* version: 20190505b
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

    plotInterpFlag = False
    
    setRange = True
    rangeMin = 600
    rangeMax = 700

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 InterpXRange <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    TFile = os.path.splitext(sys.argv[1])[0]
    if dP.setRange:
        TFile += '_Interp'+str(dP.rangeMin)+'-'+str(dP.rangeMax)+'.csv'
    else:
        TFile += '_Interp.csv'
    dfP = readParamFile(sys.argv[1])

    dfPint = interpolate(dfP)

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
def interpolate(dfP):
    dfPint = dfP.copy()
    from scipy import interpolate
    
    x0 = dfP.index.values
    
    #for i in range(len(dfP.columns)):
    for i in np.arange(1,len(dfP.columns),2):
        print(i)
        x1 = dfP.iloc[:,i].values
        y1 = dfP.iloc[:,1+i].values
        print(x1, y1)
        f1 = interpolate.interp1d(x1,y1,fill_value="extrapolate")
        y1new = f1(x0)
        print(y1new)
        dfPint.iloc[:,i] = x0
        dfPint.iloc[:,1+i] = y1new
        print(dfPint)
    
        if dP.plotInterpFlag:
            plotInterp(x0,x1,y1,y1new)

    if dP.setRange:
        dfPint = dfPint[(dfPint.index.values >= int(dP.rangeMin)) & (dfPint.index.values<int(dP.rangeMax))]
    
    for i in range(int(len(dfP.columns)/2)):
        dfPint.drop(columns=dfPint.columns[i+1], axis=1,inplace=True)
    print(dfPint)
    return dfPint


#************************************
# Plot Interpolated/Original spectra
#************************************
def plotInterp(x0,x1,y1,y1new):
    import matplotlib.pyplot as plt
    plt.plot(x0,y1new,'r', label='interp/extrap')
    plt.plot(x1,y1, 'b--', label='data')
    if dP.setRange:
        plt.xlim([dP.rangeMin, dP.rangeMax])
    plt.legend()
    plt.show()

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
