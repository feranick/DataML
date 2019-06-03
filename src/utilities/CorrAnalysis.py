#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* CorrAnalysis
* Correlation analysis
*
* version: 20190604a
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
    
    #trainCol = [7,54]
    #predCol = [1,7]
    trainCol = [1,61]
    predCol = [61,92]
    valueForNan = -1
    
    graphX = [8,10,12,13,14]
    graphY = [62,69,78,79,80,81]
    
    validRows = [40,41,42,43]
    
    corrMin = .8
    corrMax = 1
    #corrMin = -1
    #corrMax = -.6

    plotCorr = True
    plotGraphs = False
    plotGraphsThreshold = True
    plotValidData = True
    plotLinRegression = True
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

    if dP.plotCorr:
        print(" Correlation charts saved in:",plotFile,"\n")
        plotCorrelations(dfPearson, "PearsonR_correlation", pdf)
        plotCorrelations(dfSpearman, "SpearmanR_correlation", pdf)

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
            P = dfP.iloc[:,range(lims[0],lims[1])].to_numpy()
            P[np.isnan(P)] = dP.valueForNan

        with open(paramFile, 'r') as f:
            headP = np.genfromtxt(f, unpack = False, usecols=range(lims[0],lims[1]),
                delimiter = ',', skip_header=dP.skipHeadRows, skip_footer=P.shape[0], dtype=np.str)

        print(P)
        print(headP)
        print(dfP)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return P, headP, dfP

#************************************
# Plot Correlations
#************************************
def plotCorrelations(dfP, title, pdf):
    data = dfP.to_numpy()
    Rlabels = dfP.index.tolist()
    Clabels = dfP.columns.values

    fig, ax = plt.subplots(figsize=(18, 7))
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
            if dP.plotLinRegression:
                #z = np.polyfit(x, y, dP.polyDegree, full=True)
                #print(z)
                plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, dP.polyDegree))(np.unique(x)))
                #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                #plt.plot(np.unique(x),slope*np.unique(x)+intercept)
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