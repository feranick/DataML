#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* CorrAnalysis
* Correlation analysis
*
* version: 20190320b
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py, pickle
from random import uniform
from bisect import bisect_left
from scipy.stats import pearsonr, spearmanr
from libDataML import *

#************************************
# Parameters definition
#************************************
class dP:
    
    numHeadRows = 1
    
    trainCol = [1,61]
    predCol = [61,92]
    valueForNan = -1

    plotCorr = True

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 CorrAnalysis <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    

    P,headP = readParamFile(sys.argv[1], dP.trainCol)
    V,headV = readParamFile(sys.argv[1], dP.predCol)
    rootFile = os.path.splitext(sys.argv[1])[0]
    pearsonFile = rootFile + '-p' + str(dP.predCol[0]) + '_pearsonR.csv'
    spearmanFile = rootFile + '-p' + str(dP.predCol[0]) + '_spearmanR.csv'

    pearsonR=np.empty((V.shape[1],P.shape[1]))
    spearmanR=np.empty((V.shape[1],P.shape[1]))
    for j in range(V.shape[1]):
        for i in range(P.shape[1]):
            pearsonR[j,i], _ = pearsonr(P[:,i], V[:,j])
            spearmanR[j,i], _ = spearmanr(P[:,i], V[:,j])

    #print(pearsonR)
    dfP = pd.DataFrame(pearsonR)
    dfS = pd.DataFrame(spearmanR)
    dfP.columns = headP
    dfS.columns = headP

    for i in range(V.shape[1]):
        dfP.rename(index={i:headV[i]}, inplace=True)
        dfS.rename(index={i:headV[i]}, inplace=True)

    dfP.to_csv(pearsonFile, index=True, header=True)
    print("\n PearsonR correlation summary saved in:",pearsonFile,"\n")
    dfS.to_csv(spearmanFile, index=True, header=True)
    print(" SpearmanR correlation summary saved in:",spearmanFile,"\n")

    if dP.plotCorr:
        plotCorrelation(dfP)
        plotCorrelation(dfS)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, lims):
    try:
        with open(paramFile, 'r') as f:
            P = np.genfromtxt(f, unpack = False, usecols=range(lims[0],lims[1]),
                delimiter = ',', skip_header=dP.numHeadRows)
            P[np.isnan(P)] = dP.valueForNan

        with open(paramFile, 'r') as f:
            headP = np.genfromtxt(f, unpack = False, usecols=range(lims[0],lims[1]),
                delimiter = ',', skip_footer=P.shape[0], dtype=np.str)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return P, headP

#************************************
# Open Learning Data
#************************************
def plotCorrelation(dfP):
    import seaborn as sns
    import matplotlib.pyplot as plt
    mask = np.zeros_like(dfP, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dfP, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
