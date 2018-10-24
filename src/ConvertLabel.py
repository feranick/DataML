#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* ConvertLabel
* Convert normalized Label into actual label
*
* version: 20181024b
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle
from random import uniform

#***************************************************
''' This is needed for installation through pip '''
#***************************************************
def ConvertLabel():
    main()

#***************************************************
''' Main '''
#***************************************************
def main():

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 ConvertLabel.py <pkl file> <number to convert>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    try:
        le = pickle.loads(open(sys.argv[1], "rb").read())
        print(" Opening pkl file with normalization data:",sys.argv[1],"\n")
    except:
        print("\033[1m" + " pkl file not found \n" + "\033[0m")
        return

    print(" Normalized label:",sys.argv[2])
    print(" Actual label:",le.transform_inverse_single(float(sys.argv[2])),"\n")


#************************************
''' Normalizer '''
#************************************
class Normalizer(object):
    def __init__(self, M):
        self.M = M
        self.includeFirst = dP.normalizeLabel
        self.YnormTo = dP.YnormTo
        self.min = np.zeros([self.M.shape[1]])
        self.max = np.zeros([self.M.shape[1]])
        if self.includeFirst:
            for i in range(0,M.shape[1]):
                self.min[i] = np.amin(self.M[1:,i])
                self.max[i] = np.amax(self.M[1:,i])
        else:
            for i in range(0,M.shape[1]-1):
                self.min[i] = np.amin(self.M[1:,i+1])
                self.max[i] = np.amax(self.M[1:,i+1])
    
    def transform_matrix(self,y):
        Mn = np.copy(y)
        if self.includeFirst:
            for i in range(0,y.shape[1]):
                Mn[1:,i] = np.multiply(y[1:,i] - self.min[i], self.YnormTo/(self.max[i] - self.min[i]))
        else:
            for i in range(0,y.shape[1]-1):
                Mn[1:,i+1] = np.multiply(y[1:,i+1] - self.min[i], self.YnormTo/(self.max[i] - self.min[i]))
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
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
