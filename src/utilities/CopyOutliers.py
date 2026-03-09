#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CopyOutliers
* version: 2026.03.06.3
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import sys
import os
import h5py
import numpy as np
from libDataML import *

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    numCopies = 3
    
    # Define the percentile thresholds for both tails
    lower_percentile = 5  # Grabs the bottom ~25% (your 6 lowest points)
    upper_percentile = 95  # Grabs the top ~10% (your 2 highest points)

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python CopyOutliers.py <Learning file> <low thresold> <high threshold>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    elif len(sys.argv) < 3:
        copyOutliers(sys.argv[1], dP.lower_percentile, dP.upper_percentile)
    elif len(sys.argv) < 4:
        copyOutliers(sys.argv[1], float(sys.argv[2]), dP.upper_percentile)
    else:
        copyOutliers(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))

def copyOutliers(file, low, high):
    # Load Learning dataset (purged)
    En, M = readLearnFile(file)
    
    newFile = os.path.splitext(file)[0] + '_added'

    # 1. Calculate the dynamic thresholds for both tails
    lower_threshold = np.percentile(M[:, 0], low)
    upper_threshold = np.percentile(M[:, 0], high)
    
    print(f"\n Calculated lower {low}th percentile threshold: {lower_threshold:.5f}")
    print(f" Calculated upper {high}th percentile threshold: {upper_threshold:.5f}")

    # 2. Isolate the outliers (rows where column 0 is <= lower OR >= upper)
    outliers = M[(M[:, 0] <= lower_threshold) | (M[:, 0] >= upper_threshold)]
    
    print("\n Outliers found:")
    print(outliers)

    # 3. Duplicate those specific rows
    if outliers.shape[0] > 0:
        duplicated_outliers = np.tile(outliers, (dP.numCopies, 1))

        # 4. Append them back to the main dataset
        full_data = np.vstack((M, duplicated_outliers))
        final = np.vstack((En, full_data))
        
        print(f"\n Original matrix size: {M.shape[0]} rows")
        print(f" New anchored matrix size: {full_data.shape[0]} rows\n")

        # 5. Save the new "anchored" dataset
        saveLearnFile(final, newFile)
    else:
        print(f"\n No outliers found. No new file saved.")

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        sys.exit()

    En = M[0,:]
    A = M[1:,:]
    return En, A

#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'w') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    main()
