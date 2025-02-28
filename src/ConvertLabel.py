#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
************************************************
* ConvertLabel
* Convert normalized labels into actual labels
* version: v2025.2.27.2
* By: Nicola Ferralis <feranick@hotmail.com>
************************************************
'''
#print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle
from random import uniform
from libDataML import Normalizer

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
        print(__doc__)
        print(' Usage:\n  python3 ConvertLabel.py <pkl file> <number to convert>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    try:
        norm = pickle.loads(open(sys.argv[1], "rb").read())
        print("\n Opening pkl file with normalization data:",sys.argv[1],"\n")
    except:
        print("\033[1m" + " pkl file not found \n" + "\033[0m")
        return

    print(" Normalized label:",sys.argv[2])
    print(" Actual label:",norm.transform_inverse_single(float(sys.argv[2])),"\n")


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
