#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* CorrAnalysis
* Correlation analysis
*
* version: 20190907a
*
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
*
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
from scipy import stats
import sys, os.path, h5py, pickle
from random import uniform
from bisect import bisect_left
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#************************************
# Parameters definition
#************************************
class dP:
    skipHeadRows = 0
    
    #trainCol = [3,3553]   # Raw data
    #trainCol = [3,80000]   # Raw data
    #predCol = [1,3]       # Raw data
    
    trainCol = [28,47]     # ML3
    predCol = [5,7]        # ML-3
    
    #trainCol = [7,54]
    #predCol = [1,7]
    #trainCol = [61,106]
    #predCol = [28,47]
    #predCol = [61,106]

    valueForNan = -1
    validRows = [40,41,42,43]
    
    corrMin = .8
    corrMax = 1
    #corrMin = -1
    #corrMax = -.7

    heatMapsCorr = True            # True: use for Master data
    
    plotGraphs = False
    plotGraphsThreshold = False
    plotValidData = True
    plotLinRegression = True
    graphX = [8,10,12,13,14]
    graphY = [62,69,78,79,80,81]
    
    plotCorr = False                # True: use for raw data (spectra, etc)
    stepXticksPlot = 1500
    corrSpectraFull = False
    corrSpectraMin = 0.8
    polyDegree = 1

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 CorrAnalysis <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    P,headP,dfP = readParamFile(sys.argv[1], dP.trainCol)
    V,headV,_ = readParamFile(sys.argv[1], dP.predCol)
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    pearsonFile = rootFile + '_pearsonR.csv'
    spearmanFile = rootFile + '_spearmanR.csv'
    plotFile = rootFile + '_plots.pdf'

    pearsonR=np.empty((V.shape[1],P.shape[1]))
    spearmanR=np.empty((V.shape[1],P.shape[1]))
    for j in range(V.shape[1]):
        for i in range(P.shape[1]):
            pearsonR[j,i], _ = pearsonr(P[:,i], V[:,j])
            spearmanR[j,i], _ = spearmanr(P[:,i], V[:,j])

    #print(pearsonR)
    dfPearson = pd.DataFrame(pearsonR)
    dfSpearman = pd.DataFrame(spearmanR)
    dfPearson.columns = headP
    dfSpearman.columns = headP

    for i in range(V.shape[1]):
        dfPearson.rename(index={i:headV[i]}, inplace=True)
        dfSpearman.rename(index={i:headV[i]}, inplace=True)

    dfPearson.to_csv(pearsonFile, index=True, header=True)
    print("\n PearsonR correlation summary saved in:",pearsonFile,"\n")
    dfSpearman.to_csv(spearmanFile, index=True, header=True)
    print(" SpearmanR correlation summary saved in:",spearmanFile,"\n")

    pdf = PdfPages(plotFile)

    if dP.heatMapsCorr:
        print(" Correlation heat maps saved in:",plotFile,"\n")
        heatMapsCorrelations(dfPearson, "PearsonR_correlation", pdf)
        heatMapsCorrelations(dfSpearman, "SpearmanR_correlation", pdf)
    if dP.plotCorr:
        print(" Correlation plots saved in:",plotFile,"\n")
        plotCorrelations(dfPearson, P, "PearsonR_correlation", rootFile, pdf)
        plotCorrelations(dfSpearman, P, "SpearmanR_correlation", rootFile, pdf)
        if not dP.corrSpectraFull:
            dfPearson[dfPearson<dP.corrSpectraMin] = 0
            dfSpearman[dfSpearman<dP.corrSpectraMin] = 0
            plotCorrelations(dfPearson, P, "PearsonR_correlation (Corr > "+ str(dP.corrSpectraMin)+")", rootFile, pdf)
            plotCorrelations(dfSpearman, P, "SpearmanR_correlation", rootFile, pdf)

    if dP.plotGraphs:
        num = plotGraphs(dfP, dP.graphX, dP.graphY, dP.validRows, pdf)
        print(" ",num,"Manually selected plots saved in:",plotFile,"\n")

    if dP.plotGraphsThreshold:
        num1 = plotGraphThreshold(dfP, dfPearson, dP.validRows, "PearsonR_correlation", pdf)
        num2 = plotGraphThreshold(dfP, dfPearson, dP.validRows, "SpearmanR_correlation", pdf)
        print(" ",num1+num2,"XY plots with correlation in [",dP.corrMin,",",dP.corrMax,"] saved in:",plotFile,"\n")
    pdf.close()

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, lims):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
            if lims[1]>len(dfP.columns):
                lims[1] = len(dfP.columns)
                print(" Warning: Column range is larger than actual number of column. Using full dataset")
            
            P = dfP.iloc[:,range(lims[0],lims[1])].to_numpy()
            P[np.isnan(P)] = dP.valueForNan

        with open(paramFile, 'r') as f:
            headP = np.genfromtxt(f, unpack = False, usecols=range(lims[0],lims[1]),
                delimiter = ',', skip_header=dP.skipHeadRows, skip_footer=P.shape[0], dtype=np.str)

    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    print(P.shape)
    return P, headP, dfP

#************************************
# Plot Correlations
#************************************
def plotCorrelations(dfP, P, title, filename ,pdf):
    data = dfP.to_numpy()
    
    plt.xlabel('Wavelength')
    plt.ylabel('Correlation')
    
    Rlabels = dfP.index.tolist()
    Clabels = np.float_(dfP.columns.values)
    Clabels_plot = Clabels[::dP.stepXticksPlot]
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex = True,figsize=(10, 10))
    
    ax2.set_xticks(Clabels_plot)
    ax2.set_xticklabels(Clabels_plot)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    for p in P:
        ax1.plot(Clabels, p)
    ax1.set_title(filename)
    
    for i in range(len(data)):
        ax2.plot(Clabels, data[i], label=Rlabels[i])
        ax2.set_title(title)
        ax2.legend(loc='upper left')
        pdf.savefig()
    plt.close

#************************************
# Plot Heat Maps Correlations
#************************************
def heatMapsCorrelations(dfP, title, pdf):
    data = dfP.to_numpy()
    Rlabels = dfP.index.tolist()
    Clabels = dfP.columns.values

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(data)

    ax.set_xticks(np.arange(len(Clabels)))
    ax.set_yticks(np.arange(len(Rlabels)))
    ax.set_xticklabels(Clabels)
    ax.set_yticklabels(Rlabels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    '''
    # Loop over data dimensions and create text annotations.
    for i in range(len(Rlabels)):
        for j in range(len(Clabels)):
            text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="w")
    '''

    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    cbar.ax.set_ylabel("correlation", rotation=-90, va="bottom")
    
    ax.set_title(title)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    pdf.savefig()
    #plt.savefig(title+".png", dpi = 160, format = 'png')  # Save plot
    #plt.show()
    plt.close()
    

#************************************
# Pllot Graphs based on manual input
#************************************
def plotGraphs(dfP, X, Y, validRows, pdf):
    num = 0
    for i in X:
        for j in Y:
            ylabels = dfP.columns.values[j]
            xlabels = dfP.columns.values[i]
            #plt.figure()
            plt.plot(dfP.iloc[:,i],dfP.iloc[:,j], 'bo')
            if dP.plotValidData:
                plt.plot(dfP.iloc[validRows,i].to_list(),dfP.iloc[validRows, j].to_list(), 'ro')
            plt.xlabel(xlabels)
            plt.ylabel(ylabels)
            for k, txt, in enumerate(dfP.iloc[:,0].to_list()):
                plt.annotate(txt,xy=(dfP.iloc[:,i].to_list()[k],dfP.iloc[:,j].to_list()[k]), fontsize='x-small')
            #plt.legend(loc='upper left')
            pdf.savefig()
            plt.close()
            num += 1
    return num

#************************************
# Plot Graphs based on threshold
#************************************
def plotGraphThreshold(dfP, dfC, validRows, title, pdf):
    num = 0
    #print(dfP.loc[validRows,1])
    for col in dfC.columns:
        for ind in dfC[dfC[col].between(dP.corrMin,dP.corrMax)].index:
        
            x = dfP[col].to_list()
            y = dfP[ind].to_list()
            plt.plot(x,y, 'bo')
            if dP.plotValidData:
                plt.plot(dfP.loc[validRows,col].to_list(),dfP.loc[validRows, ind].to_list(), 'ro')
            plt.xlabel(col)
            plt.ylabel(ind)
            plt.title(title+": {0:.3f}".format(dfC[col].loc[ind]))
            for k, txt, in enumerate(dfP.iloc[:,0].to_list()):
                plt.annotate(txt,xy=(x[k],y[k]), fontsize='x-small')
            if dP.plotLinRegression and col != ind:
                #z = np.polyfit(x, y, dP.polyDegree, full=True)
                #print(z)
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    plt.text(min(x), max(y),"{0:s} = {2:.3f}*{1:s} + {3:.3f}".format(ind, col, slope, intercept))
                    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, dP.polyDegree))(np.unique(x)))
                except:
                    pass
            #plt.legend(loc='upper left')
            pdf.savefig()
            plt.close()
            num+=1
    return num

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
