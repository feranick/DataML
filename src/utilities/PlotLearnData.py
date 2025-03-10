#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* PlotLearnData
* Plot learning data
* version: 2025.03.07.1
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
    excludeZeroFeatures = False
    normalize = False
    
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
    plotAugmData(A1.shape, A1, A2, A3, rootFile+"_plots.pdf")
 
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

    if dP.normalize:
        print("  Normalization of feature matrix to 1")
        if newNorm:
            print("  Normalization parameters saved in:", dP.norm_file,"\n")
            norm = Normalizer(M, dP)
        else:
            print("  Normalization parameters from:", dP.norm_file,"\n")
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
        M = norm.transform(M)
        norm.save()
    
    if dP.excludeZeroFeatures:
        ind = np.any(M == 0, axis=1)
        ind[0] = False
        M = M[~ind]
    
    En = M[0,:]
    A = M[1:,:]
    Cl = M[1:,0]
    
    return En, A, M

#************************************
# Plot augmented training data
#************************************
def plotAugmData(shape, A1,A2,A3, plotFile):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf = PdfPages(plotFile)
    
    for i in range(1, shape[1]):
        xA1 = A1[:,0]
        yA1 = A1[:,i]
        xA2 = A2[:,0]
        yA2 = A2[:,i]
        if A3 is not None:
            xA3 = A3[:,0]
            yA3 = A3[:,i]
            plt.plot(xA3,yA3, 'gx', markersize=3, mfc='none')
        plt.plot(xA1,yA1, 'bo', markersize=3)
        plt.plot(xA2,yA2, 'ro', markersize=3)
        plt.xlabel("col "+str(i))
        plt.ylabel("col 0")
        pdf.savefig()
        plt.close()
    pdf.close()
    print(" Plots saved in:", plotFile,"\n")
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
