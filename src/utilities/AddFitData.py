#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Add Fit Data
* version: 20210606a
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
    #cOffset = [6,6,6,6,6,6,6,6,6,22,6,22,0]
    #cOffset = [0,0,0,0,0,0,0,0,0,50,0,0,0]
    cOffset = [6,6,6,6,6,6,6,6,6,22,6,22,0]
    
    useC9 = False
    
    # z = a1*x + a2*y + a3*x*y + a4*x*x + a5*y*y + c
    # P1-P2-P3
    a1 = 0.008915138339267296
    a2 = 0.010368946821814518
    a3 = -0.00017088314033818566
    a4 = -2.883568103373771e-06
    a5 = -0.00044561640223113823
    a =  -0.008652904655741422
    
    #P1-P3-P2
    b1 = 0.3259918685462891
    b2 = -22.484345206709786
    b3 = -0.27406242654845586
    b4 = 0.00017125686861163558
    b5 = 25.907094484839803
    b =  0.5266619381212134
    
    # P1-C9-P3
    c1 = 0.008821312009279216
    c2 = -0.043404469072183996
    c3 = -7.032782207110197e-05
    c4 = -8.29818533720078e-06
    c5 = 0.0017265086355506612
    c =  0.0051781559844021885
    
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
    
    #print(dfP.iloc[:,1:11])
    #print(dfP.iloc[:,11:13])
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    if dP.useC9:
        noisyFile = rootFile + '_noisyFitC9-' + sys.argv[2]
        print(" P1, C9 -> P3; P1, P3 -> P2")
    else:
        noisyFile = rootFile + '_noisyFit-' + sys.argv[2]
        print(" P1, P2 -> P3")
    
    if dP.customOffset == True:
        noisyFile += 'cust.csv'
        offs = dP.cOffset
    else:
        noisyFile += 'opcFit'+sys.argv[3]+'.csv'
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
    #print(dfP)
    dfP_temp = dfP.copy()
    dfP_noise = dfP.copy()
    for i in range(1, num):
        #print(dfP.iloc[:,1:9])
        #print(dfP.iloc[:,9:13])
        factor = (offset*np.random.uniform(-0.01,0.01,(dfP_temp.iloc[:,1:].shape)))
        
        #print("Before 1",dfP.iloc[:,8:13])
        dfP_temp.iloc[:,1:] = dfP.iloc[:,1:].mul(1+factor)
        
        #print("C9",dfP.iloc[:,9])
        #print("P1",dfP.iloc[:,10])
        #print("P2",dfP.iloc[:,11])
        #print("P3",dfP.iloc[:,12])
        
        #print("Before 2",dfP_temp.iloc[:,8:13])
        if dP.useC9:
            dfP_temp.iloc[:,12] = p1c9p3(dfP_temp.iloc[:,10], dfP_temp.iloc[:,9])
            dfP_temp.iloc[:,11] = p1p3p2(dfP_temp.iloc[:,10], dfP_temp.iloc[:,12])
        else:
            dfP_temp.iloc[:,12] = p1p2p3(dfP_temp.iloc[:,10], dfP_temp.iloc[:,11])
        
        #print("After",dfP_temp.iloc[:,8:13])

        dfP_noise = dfP_noise.append(dfP_temp, ignore_index=True)
        
    #print(dfP_noise[dfP_noise["Specimen"] == "2194"])
    return dfP_noise
    
def p1p2p3(x,y):
    z = dP.a1*x + dP.a2*y + dP.a3*x*y + dP.a4*x*x + dP.a5*y*y + dP.a
    return z
    
def p1p3p2(x,y):
    z = dP.b1*x + dP.b2*y + dP.b3*x*y + dP.b4*x*x + dP.b5*y*y + dP.b
    return z
    
def p1c9p3(x,y):
    z = dP.c1*x + dP.c2*y + dP.c3*x*y + dP.c4*x*x + dP.c5*y*y + dP.c
    return z

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
