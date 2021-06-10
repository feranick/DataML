#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Add Fit Data
* version: 20210610a
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
    
    useNormal = False
    normStDev = 0.0025
    unifStDev = 0.01
    postFitRand = True
    
    useP1P2P3 = True
    useP1P3P2 = False
    useP2P3P1 = False
    useP1C9P3 = False
    
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
    
    #P2-P3-P1
    c1 = -3.3985989701486874
    c2 = 64.37435919405675
    c3 = 8.611010708553058
    c4 = 0.10011439298175162
    c5 = -6.944056398339368
    c =  2.236494642942631

    # P1-C9-P3
    d1 = 0.008821312009279216
    d2 = -0.043404469072183996
    d3 = -7.032782207110197e-05
    d4 = -8.29818533720078e-06
    d5 = 0.0017265086355506612
    d =  0.0051781559844021885
    
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 4 and dP.customOffset == False:
        print(' Usage:\n  python3 AddFitData <paramFile> <#additions>')
        print('  Offset: multiplier in units of percent (i.e. 1 is 1%)')
        print('  Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dfP = readParamFile(sys.argv[1])
    noisyFile = os.path.splitext(sys.argv[1])[0]
    
    if dP.customOffset == True:
        offs = dP.cOffset
    else:
        offs = int(sys.argv[3])
        
    dfP_final = dfP.copy()
        
    if dP.useP1P2P3:
        noisyFile += '_noisyFitP1P2P3-' + sys.argv[2]
        print(" P1, P2 -> P3")
        dfP_final = addAugData(dfP, dfP_final, int(sys.argv[2]), offs)
        
    if dP.useP1P3P2:
        noisyFile += '_noisyFitP1P3P2-' + sys.argv[2]
        print(" P1, P3 -> P2")
        dfP_final = addAugData(dfP, dfP_final, int(sys.argv[2]), offs)
        
    if dP.useP2P3P1:
        noisyFile += '_noisyFitP2P3P1-' + sys.argv[2]
        print(" P2, P3 -> P1")
        dfP_final = addAugData(dfP, dfP_final, int(sys.argv[2]), offs)
        
    if dP.useP1C9P3:
        noisyFile += '_noisyFitP1C9P3-' + sys.argv[2]
        print(" P1, C9 -> P3; P1, P3 -> P2")
        dfP_final = addAugData(dfP, dfP_final, int(sys.argv[2]), offs)
        
    if dP.useNormal:
        noisyFile += 'normal' + str(dP.normStDev)
    else:
        noisyFile += 'uniform' + str(dP.unifStDev)
        
    if dP.postFitRand:
        noisyFile += '_postFitRand'

    if dP.customOffset == True:
        noisyFile += '_cust'+str(dP.cOffset[0])+'.csv'
    else:
        noisyFile += '_opcFit'+sys.argv[3]+'.csv'

    #dfP_noise = addAugData(dfP, int(sys.argv[2]), offs)
    dfP_final.to_csv(noisyFile, index=False, header=True)
    
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
# Augment Data
#************************************
def addAugData(dfP, dfP_final, num, offset):
    dfP_temp = dfP.copy()
    dfP_noise = pd.DataFrame(columns=dfP.columns)
    
    for i in range(1, num+1):
        #factor = (offset*np.random.uniform(-0.01,0.01,(dfP_temp.iloc[:,1:].shape)))
        if dP.useNormal:
            factor = offset*np.random.normal(0,dP.normStDev,(dfP_temp.iloc[:,1:].shape))
        else:
            factor = offset*np.random.uniform(-dP.unifStDev,dP.unifStDev,(dfP_temp.iloc[:,1:].shape))
            
        dfP_temp.iloc[:,1:] = dfP.iloc[:,1:].mul(1+factor)
        
        #print("C9",dfP.iloc[:,9])
        #print("P1",dfP.iloc[:,10])
        #print("P2",dfP.iloc[:,11])
        #print("P3",dfP.iloc[:,12])
        
        if dP.useP1P2P3:
            dfP_temp.iloc[:,12] = p1p2p3(dfP_temp.iloc[:,10], dfP_temp.iloc[:,11])
            if dP.postFitRand:
                dfP_temp.iloc[:,12] = dfP_temp.iloc[:,12].mul(1+factor[:,11])
            
        if dP.useP1P3P2:
            dfP_temp.iloc[:,11] = p1p3p2(dfP_temp.iloc[:,10], dfP_temp.iloc[:,12])
            if dP.postFitRand:
                dfP_temp.iloc[:,11] = dfP_temp.iloc[:,11].mul(1+factor[:,10])
                
        if dP.useP2P3P1:
            dfP_temp.iloc[:,10] = p2p3p1(dfP_temp.iloc[:,11], dfP_temp.iloc[:,12])
            if dP.postFitRand:
                dfP_temp.iloc[:,10] = dfP_temp.iloc[:,10].mul(1+factor[:,9])
        
        if dP.useP1C9P3:
            dfP_temp.iloc[:,12] = p1c9p3(dfP_temp.iloc[:,10], dfP_temp.iloc[:,9])
            if dP.postFitRand:
                dfP_temp.iloc[:,12] = dfP_temp.iloc[:,12].mul(1+factor[:,11])
            dfP_temp.iloc[:,11] = p1p3p2(dfP_temp.iloc[:,10], dfP_temp.iloc[:,12])
            if dP.postFitRand:
                dfP_temp.iloc[:,11] = dfP_temp.iloc[:,11].mul(1+factor[:,10])
                
        dfP_noise = dfP_noise.append(dfP_temp, ignore_index=True)
    dfP_final = dfP_final.append(dfP_noise, ignore_index=True)
    
    return dfP_final

#************************************
# Fitting methods
#************************************
def p1p2p3(x,y):
    z = dP.a1*x + dP.a2*y + dP.a3*x*y + dP.a4*x*x + dP.a5*y*y + dP.a
    #z = np.multiply(dP.a1, x) + np.multiply(dP.a2,y) + np.multiply(dP.a3,np.matmul(x,y)) + np.multiply(dP.a4,np.matmul(x,x)) + np.multiply(dP.a5,np.matmul(y,y)) + dP.a
    #z = np.add(np.add(np.add(np.add(np.multiply(dP.a1, x), np.multiply(dP.a2,y)), np.multiply(dP.a3,np.matmul(x,y))), np.multiply(dP.a4,np.matmul(x,x))), np.multiply(dP.a5,np.matmul(y,y))) + dP.a
    return z
    
def p1p3p2(x,y):
    z = dP.b1*x + dP.b2*y + dP.b3*x*y + dP.b4*x*x + dP.b5*y*y + dP.b
    return z
    
def p2p3p1(x,y):
    z = dP.c1*x + dP.c2*y + dP.c3*x*y + dP.c4*x*x + dP.c5*y*y + dP.c
    return z
    
def p1c9p3(x,y):
    z = dP.d1*x + dP.d2*y + dP.d3*x*y + dP.d4*x*x + dP.d5*y*y + dP.d
    return z

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
