#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* ConvertLabel
* Convert normalized labels into actual labels
*
* version: 20181026a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle
from random import uniform
from bisect import bisect_left

#***************************************************
''' This is needed for installation through pip '''
#***************************************************
def ConvertLabel():
    main()

#************************************
''' Main '''
#************************************
def main():

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 ConvertLabel.py <pkl file> <number to convert>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    try:
        norm = pickle.loads(open(sys.argv[1], "rb").read())
    print(" Opening pkl file with normalization data:",sys.argv[1],"\n")
    except:
        print("\033[1m" + " pkl file not found \n" + "\033[0m")
        return

    print(" Normalized label:",sys.argv[2])
    print(" Actual label:",norm.transform_inverse_single(float(sys.argv[2])),"\n")


#************************************
''' Normalizer '''
#************************************
class Normalizer(object):
    def __init__(self, M):
        self.M = M
        self.includeFirst = dP.normalizeLabel
        self.useCustomRound = dP.useCustomRound
        self.YnormTo = dP.YnormTo
        self.stepNormLabel = dP.stepNormLabel
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])
        
        self.data = np.arange(0,1,self.stepNormLabel)
        
        if self.includeFirst:
            self.min[0] = np.amin(self.M[1:,0])
            self.max[0] = np.amax(self.M[1:,0])
        
        for i in range(1,M.shape[1]):
            self.min[i] = np.amin(self.M[1:,i])
            self.max[i] = np.amax(self.M[1:,i])
    
    def transform_matrix(self,y):
        Mn = np.copy(y)
        if self.includeFirst:
            Mn[1:,0] = np.multiply(y[1:,0] - self.min[0], self.YnormTo/(self.max[0] - self.min[0]))
            if self.useCustomRound:
                customData = CustomRound(self.data)
                for i in range(1,y.shape[0]):
                    Mn[i,0] = customData(Mn[i,0])

        for i in range(1,y.shape[1]):
            Mn[1:,i] = np.multiply(y[1:,i] - self.min[i], self.YnormTo/(self.max[i] - self.min[i]))
        return Mn
    
    def transform_valid(self,V):
        Vn = np.copy(V)
        for i in range(0,V.shape[0]):
            Vn[i,1] = np.multiply(V[i,1] - self.min[i], self.YnormTo/(self.max[i] - self.min[i]))
        return Vn
    
    def transform_inverse_single(self,v):
        vn = self.min[0] + v*(self.max[0] - self.min[0])/self.YnormTo
        return vn

    def save(self, name):
        with open(name, 'ab') as f:
            f.write(pickle.dumps(self))

#************************************
''' CustomRound '''
#************************************
class CustomRound:
    def __init__(self,iterable):
        self.data = sorted(iterable)

    def __call__(self,x):
        data = self.data
        ndata = len(data)
        idx = bisect_left(data,x)
        if idx <= 0:
            return data[0]
        elif idx >= ndata:
            return data[ndata-1]
        x0 = data[idx-1]
        x1 = data[idx]
        if abs(x-x0) < abs(x-x1):
            return x0
        return x1

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
