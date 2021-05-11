#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* MasterDataMaker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: 20191113a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
import pandas as pd
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************
class defParam:
    
    skipHeadRows = 0
    valueForNan = -1

    # set boundaries intensities for when to
    # fill in in absence of data
    leftBoundary = 0
    rightBoundary = 0
    
    # set to True to set boundaries as the min
    # values for intensities when to
    # fill in in absence of data
    useMinForBoundary = False

def main():
    if len(sys.argv) < 5:
        enInit = 176
        enFin = 194
        enStep = 2
    else:
        enInit = sys.argv[2]
        enFin =  sys.argv[3]
        enStep = sys.argv[4]
        if len(sys.argv) < 6:
            threshold = 0
        else:
            threshold = sys.argv[5]

    if len(sys.argv) == 7:
        defParam.useMinForBoundary = True
    
    #try:
    processMultiFile(sys.argv[1], enInit, enFin, enStep, threshold)
    #except:
    #    usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(masterFile, enInit, enFin, enStep, threshold):
    size = 0
    compound=[]
    masterFileRoot = os.path.splitext(masterFile)[0]
    masterFileExt = os.path.splitext(masterFile)[1]

    #summary_filename = masterFileRoot + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    #summary = "file"+",enInit="+str(enInit)+",enFin="+str(enFin)+",enStep="+str(enStep)+\
    #        ",threshold="+str(threshold)+"\n"
    
    # Read, if exisiting, learnFile
    if os.path.exists(masterFile):
        print('\n\033[1m' + ' Train data file found. Opening...' + '\033[0m')
        dfP = readMasterFile(masterFile)
        print(dfP)
        
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        dfP = pd.DataFrame(columns=np.arange(float(enInit), float(enFin), float(enStep), dtype=np.float))
        dfP.insert(0,"file",'0')
        #dfP.loc[0] = np.arange(float(enInit), float(enFin), float(enStep), dtype=np.float)
        #dfP = pd.DataFrame(np.arange(float(enInit), float(enFin), float(enStep), dtype=np.float))
        print(dfP)


    # process sample data
    for ind, f in enumerate(sorted(os.listdir("."))):
        if (f is not masterFile and os.path.splitext(f)[-1] is ".txt"):
            dfP = makeFile(f, dfP, threshold)

    print('\n Energy scale: [', str(enInit),',',
            str(enFin), ']; Step:', str(enStep),
            '; Threshold:', str(threshold),'\n')
            
    print(dfP)

    dfP.to_csv(os.path.splitext(masterFile)[0]+'.csv', index=False)

#**********************************************
''' Add data to Training file '''
#**********************************************
def makeFile(sampleFile, dfP, threshold):
    EnT = dfP.columns.values[1:].astype(float)
    M = dfP.values
    print('\n Process file : ' + sampleFile)
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = '\t', skiprows = 0)
            if(En.size == 0):
                print('\n Empty file \n' )
                return dfP
        with open(sampleFile, 'r') as f:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = '\t', skiprows = 0)
    
        R[R<float(threshold)*np.amax(R)/100] = 0
        print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
        print(' Setting datapoints below ', threshold, '% of max (',str(np.amax(R)),')')
    except:
        print('\033[1m' + sampleFile + ' file not found \n' + '\033[0m')
        return dfP

    if EnT.shape[0] == En.shape[0]:
        print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
    else:
        print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
        if defParam.useMinForBoundary:
            print(" Boundaries: Filling in with min values")
            defParam.leftBoundary = R[0]
            defParam.rightBoundary = R[R.shape[0]-1]
        else:
            print(" Boundaries: Filling in preset values")
        print("  Left:",defParam.leftBoundary,"; Right:",defParam.leftBoundary)
        
        R = np.interp(EnT, En, R, left = defParam.leftBoundary, right = defParam.rightBoundary)
        print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')
    

    dfP.loc[len(dfP),1:] = R
    dfP['file'].iloc[-1] = sampleFile

    return dfP

#************************************
''' Open Master File '''
#************************************
def readMasterFile(masterFile):
    print(" Opening learning file: "+masterFile+"\n")
    try:
        with open(masterFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",",
                skiprows=defParam.skipHeadRows, na_values=defParam.valueForNan)

    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    return dfP

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 MasterDatamaker.py <masterfile> <enInitial> <enFinal> <enStep> <threshold> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
