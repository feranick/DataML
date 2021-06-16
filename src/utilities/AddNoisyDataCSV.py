#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Add Noisy Data to CSV
* version: 20210616d
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py
from random import uniform

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
    customOffset = True
    cOffset = [6,6,6,6,6,6,6,6,6,22,6,22,0]
    #cOffset = [6,4,3,2,2,1,6,2,3,22,6,22,0]

    useNormal = True
    normStDev = 0.0025
    unifStDev = 0.01
    
    mulFactor = True
    addFactor = False
    
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
    noisyFile = rootFile + '_noisy-' + sys.argv[2]
    
    if dP.useNormal:
        noisyFile += 'normal' + str(dP.normStDev)
    else:
        noisyFile += 'uniform' + str(dP.unifStDev)
        
    if dP.mulFactor:
        noisyFile += '-M'
    if dP.addFactor:
        noisyFile += '-A'
    
    if dP.customOffset:
        noisyFile += '_cust'
        for i in dP.cOffset:
            noisyFile += '-'+str(i)
        noisyFile +='.csv'
        offs = dP.cOffset
    else:
        noisyFile += '_opc'+sys.argv[3]+'.csv'
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
        if dP.useNormal:
            factor = offset*np.random.normal(0,dP.normStDev,(dfP_temp.iloc[:,1:].shape))
        else:
            factor = offset*np.random.uniform(-dP.unifStDev,dP.unifStDev,(dfP_temp.iloc[:,1:].shape))

        if dP.mulFactor:
            dfP_temp.iloc[:,1:] = dfP.iloc[:,1:].mul(1+factor)
        if dP.addFactor:
            dfP_temp.iloc[:,1:] = dfP_temp.iloc[:,1:].add(factor)
        
        dfP_noise = dfP_noise.append(dfP_temp, ignore_index=True)
        
    #print(dfP_noise[dfP_noise["Specimen"] == "2194"])
    return dfP_noise

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
