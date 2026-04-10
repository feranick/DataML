#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CopyOutliers
* version: 2016.04.10.2
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
    plotData = True
    useThresholdOnly = False
    
    # Define the percentile thresholds for both tails
    lower_percentile = 10  # Grabs the bottom ~10%
    upper_percentile = 90  # Grabs the top ~10%

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
    En, A, M = readLearnFile(file)
    
    if dP.useThresholdOnly:
        print("  --- Univariate---")
        outliers = copyOutliersUnivariate(A, low, high)
    else:
        print("  --- Multivariate---")
        outliers = copyOutliersMultivariate(A, low, high)
    
    # 3. Duplicate those specific rows
    if outliers.shape[0] > 0:
        duplicated_outliers = np.tile(outliers, (dP.numCopies, 1))

        # 4. Append them back to the main dataset
        full_data = np.vstack((A, duplicated_outliers))
        final = np.vstack((En, full_data))
        
        print(f"\n Original matrix size: {M.shape[0]} rows")
        print(f" New anchored matrix size: {full_data.shape[0]} rows\n")

        newFile = os.path.splitext(file)[0] + '_added'
        if dP.plotData:
            plotOutliers(dP, A.shape, A, outliers, newFile+"_copied-outliers-plots.pdf")
            
        # 5. Save the new "anchored" dataset
        saveLearnFile(final, newFile)
    else:
        print(f"\n No outliers found. No new file saved.")

# Use univariate (for the extremes) and multivariate (for outliers) analysis to replicate data.
def copyOutliersMultivariate(A, low, high):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import NearestNeighbors

    # --- Step 1: Univariate (Target-Based) Outliers ---
    # Calculate the dynamic thresholds for both tails of column 0
    lower_threshold = np.percentile(A[:, 0], low)
    upper_threshold = np.percentile(A[:, 0], high)
    
    print(f"\n Calculated lower {low}th percentile target threshold: {lower_threshold:.5f}")
    print(f" Calculated upper {high}th percentile target threshold: {upper_threshold:.5f}")

    # Create a boolean mask for univariate outliers
    univariate_mask = (A[:, 0] <= lower_threshold) | (A[:, 0] >= upper_threshold)

    # --- Step 2: Multivariate (Distance-Based) Outliers ---
    # Normalize all features and target to [0, 1] scale for accurate distance calculation
    scaler = MinMaxScaler()
    A_normalized = scaler.fit_transform(A)

    # Calculate Euclidean distance to the nearest neighbors (e.g., 5 neighbors)
    k_neighbors = min(6, A_normalized.shape[0]) # 6 to include the point itself + 5 neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='euclidean').fit(A_normalized)
    distances, indices = nbrs.kneighbors(A_normalized)

    # Calculate average distance to those N neighbors (excluding the point itself at index 0)
    avg_distances = distances[:, 1:].mean(axis=1)

    # Determine distance threshold (using the 'high' percentile, e.g., top 5% most distant points)
    multivariate_threshold = np.percentile(avg_distances, high)
    
    print(f" Calculated multivariate distance threshold ({high}th percentile): {multivariate_threshold:.5f}")

    # Create a boolean mask for multivariate outliers
    multivariate_mask = (avg_distances >= multivariate_threshold)

    # --- Step 3: Combine and Extract ---
    # Union of both masks: A point is an outlier if it fails EITHER test
    combined_mask = univariate_mask | multivariate_mask
    outliers = A[combined_mask]
    
    print(f"\n Summary:")
    print(f" Target-based outliers found: {np.sum(univariate_mask)}")
    print(f" Multivariate outliers found: {np.sum(multivariate_mask)}")
    print(f" Total unique outliers isolated: {outliers.shape[0]}")
    print(outliers)
    return outliers

def copyOutliersUnivariate(A, low, high):
    # 1. Calculate the dynamic thresholds for both tails
    lower_threshold = np.percentile(A[:, 0], low)
    upper_threshold = np.percentile(A[:, 0], high)
    
    print(f"\n Calculated lower {low}th percentile threshold: {lower_threshold:.5f}")
    print(f" Calculated upper {high}th percentile threshold: {upper_threshold:.5f}")

    # 2. Isolate the outliers (rows where column 0 is <= lower OR >= upper)
    outliers = A[(A[:, 0] <= lower_threshold) | (A[:, 0] >= upper_threshold)]
    
    print("\n Outliers found:")
    print(outliers)
    return outliers

#************************************
# Plot outliers
#************************************
def plotOutliers(dP, shape, A1, A2, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf = PdfPages(plotFile)
        
    for i in range(1, shape[1]):
        xA1 = A1[:,i]
        yA1 = A1[:,0]
        xA2 = A2[:,i]
        yA2 = A2[:,0]
        
        #xA1,yA1 = removeZeros(dP, xA1,yA1)
        #xA2,yA2 = removeZeros(dP, xA2,yA2)
        plt.plot(xA1,yA1, 'bo', markersize=3)
        plt.plot(xA2,yA2, 'ro', markersize=3)
        plt.xlabel("col "+str(i)+" - feature parameter")
        plt.ylabel("col 0 - predicted parameter")
        pdf.savefig()
        plt.close()
    pdf.close()
    print(" Plots saved in:", plotFile, "\n")

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
    return En, A, M

#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        print(" New training file (txt) saved in:", learnFile+"\n")
        with open(learnFile, 'w') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" New training file (hdf5) saved in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    main()
