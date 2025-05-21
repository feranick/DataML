#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* PlotLearnData
* Plot learning data
* version: 2025.05.16.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle, configparser
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def PlotLearnData():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    excludeZeroFeatures = True
    useLabels = False
    labels = [8,4,5,35]  # First parameter is always the performance
    
#************************************
# Main
#************************************
def main():
    dP = Conf()
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 PlotLearnData.py <learnData> <validData> <AugmentedLearnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En1, A1, M1 = readLearnFile(dP, sys.argv[1], True)
    En2, A2, M2 = readLearnFile(dP, sys.argv[2], True)
    if len(sys.argv) == 4:
        En3, A3, M3 = readLearnFile(dP, sys.argv[3], True)
    else:
        A3 = None
    
    rootFile = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    plotAugmData(dP, A1.shape, A1, A2, A3, rootFile+"_learn-plots.pdf")
 
#************************************
# Open Learning Data
#************************************
def readLearnFile(dP, learnFile, newNorm):
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
        return

    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#************************************
# Plot augmented training data
#************************************
def plotAugmData(dP, shape, A1,A2,A3, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf = PdfPages(plotFile)
        
    for i in range(1, shape[1]):
        xA1 = A1[:,i]
        yA1 = A1[:,0]
        xA2 = A2[:,i]
        yA2 = A2[:,0]
        if A3 is not None:
            xA3 = A3[:,0]
            yA3 = A3[:,i]
            xA3,yA3 = removeZeros(dP, xA3,yA3)
            plt.plot(xA3,yA3, 'gx', markersize=3, mfc='none')
        
        xA1,yA1 = removeZeros(dP, xA1,yA1)
        xA2,yA2 = removeZeros(dP, xA2,yA2)
        plt.plot(xA1,yA1, 'bo', markersize=3)
        plt.plot(xA2,yA2, 'ro', markersize=3)
        if dP.useLabels:
            plt.xlabel("m"+str(dP.labels[i]))
            plt.ylabel("Perf"+str(dP.labels[0]))
        else:
            plt.xlabel("col "+str(i)+" - feature parameter")
            plt.ylabel("col 0 - predicted parameter")
        pdf.savefig()
        plt.close()
    pdf.close()
    print(" Plots saved in:", plotFile,"\n")
    
def removeZeros(dP, X_in, Y_in):
    if dP.excludeZeroFeatures:
        X = np.asarray(X_in) # Ensure numpy array for boolean indexing
        Y = np.asarray(Y_in)
        # Create a mask for elements to keep: X != 0 AND Y != 0
        keep_mask = (X != 0) & (Y != 0)
        return X[keep_mask], Y[keep_mask]
    return X_in, Y_in # Return original if not excluding
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
