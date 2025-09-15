#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CorrAnalysis
* Correlation Analysis
* version: 2025.09.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
from scipy import stats
import sys, os.path, configparser, ast
from datetime import datetime, date
from scipy.stats import pearsonr, spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from numpy.polynomial.polynomial import Polynomial as polyfit

#***************************************************
# This is needed for installation through pip
#***************************************************
def CorrAnalysis():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    def __init__(self):
        self.appName = "CorrAnalysis"
        confFileName = "CorrAnalysis.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)        
        self.skipHeadColumns = self.skipHeadColumns-1
        
        if self.specifyColumns == False:
            self.trainCol = [item for item in range(self.trainCol[0]+self.skipHeadColumns, self.trainCol[1]+1+self.skipHeadColumns)]
            if len(self.predCol)!=1:
                self.predCol = [item for item in range(self.predCol[0]+self.skipHeadColumns, self.predCol[1]+1+self.skipHeadColumns)]
            else:
                self.predCol = [self.predCol[0] + self.skipHeadColumns]
        else:
            self.trainCol = [x + self.skipHeadColumns for x in self.trainCol]
            if len(self.predCol)!=1:
                self.predCol = [x + self.skipHeadColumns for x in self.predCol]
            else:
                self.predCol = [self.predCol[0] + self.skipHeadColumns]
    
        if self.includeAdditionalCol == True:
            self.inTrainCol=self.trainCol[-1]
            self.trainCol.extend(list(item for item in range(self.initialAdditionalCol, self.finalAdditionalCol+1)))
            self.predCol = [item for item in range(self.predCol[0]+self.trainCol[-1]-self.inTrainCol, self.predCol[1]+self.trainCol[-1]-self.inTrainCol+1)]
            
        if self.plotValidData:
            self.validRows = [x - 1 for x in self.validRows]
            
        self.cmap = 'plasma'
        #self.cmap = 'tab20'
                                    
    def corrAnalysisDef(self):
        self.conf['Parameters'] = {
            'skipHeadRows' : 4,
            'skipHeadColumns' : 7,
            'skipEmptyColumns' : True,
            
            'specifyColumns'  : True,
            'trainCol' : [2,113],
            'predCol' : [114,121],
    
            'includeAdditionalCol' : False,
            'initialAdditionalCol' : 41,
            'finalAdditionalCol' : 95,
    
            'separateValidFile' : False,
            'validRows' : [0],   # ORNL

            'valueForNan' : 0,
            'removeNaNfromCorr' : False,

            ### Heat Maps
            'heatMapsCorr' : True,             # True: use for Master data
            'heatMapCorrFull' : False,          #True: plot all correlation data
            'corrMax' : 1,
            'corrMin' : 0.75,
    
            ### Plotting correlation 2D plots
            'plotSelectedGraphs' : False,
            'plotGraphsThreshold' : True,
            'graphX' : [1,2],
            'graphY' : [3,4],
            'plotValidData' : False,
            'plotLinRegression' : True,
            'addSampleTagPlot' : True,
            'polyDegree' : 1,
            
            ### Enable s[ecial coloring in plots
            'plotSpecificColors' : False,
            'columnSpecColors' : 89,
            'customColors' : False,
            'custColorsList' : ['blue','grey', 'black', 'red', 'yellow'],
    
            ### Plotting Spectral correlations
            'plotSpectralCorr' : False,                # True: use for raw data (spectra, etc)
            'stepXticksPlot' : 1500,
            'corrSpectraFull' : False,
            'corrSpectraMin' : 0.8,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.corrAnalysisPar = self.conf['Parameters']

            self.skipHeadRows = self.conf.getint('Parameters','skipHeadRows')
            self.skipHeadColumns = self.conf.getint('Parameters','skipHeadColumns')
            self.skipEmptyColumns = self.conf.getboolean('Parameters','skipEmptyColumns')
            self.specifyColumns = self.conf.getboolean('Parameters','specifyColumns')
            self.trainCol = ast.literal_eval(self.corrAnalysisPar['trainCol'])
            self.predCol = ast.literal_eval(self.corrAnalysisPar['predCol'])
            
            self.includeAdditionalCol = self.conf.getboolean('Parameters','includeAdditionalCol')
            self.initialAdditionalCol = self.conf.getint('Parameters','initialAdditionalCol')
            self.finalAdditionalCol = self.conf.getint('Parameters','finalAdditionalCol')
    
            self.separateValidFile = self.conf.getboolean('Parameters','separateValidFile')
            self.validRows = ast.literal_eval(self.corrAnalysisPar['validRows'])

            self.valueForNan = self.conf.getint('Parameters','valueForNan')
            self.removeNaNfromCorr = self.conf.getboolean('Parameters','removeNaNfromCorr')

            self.heatMapsCorr = self.conf.getboolean('Parameters','heatMapsCorr')            # True: use for Master data
            self.heatMapCorrFull = self.conf.getboolean('Parameters','heatMapCorrFull')          #True: plot all correlation data
            self.corrMax = self.conf.getfloat('Parameters','corrMax')
            self.corrMin = self.conf.getfloat('Parameters','corrMin')
    
            self.plotSelectedGraphs = self.conf.getboolean('Parameters','plotSelectedGraphs')
            self.plotGraphsThreshold = self.conf.getboolean('Parameters','plotGraphsThreshold')
            self.plotValidData = self.conf.getboolean('Parameters','plotValidData')
            self.plotLinRegression = self.conf.getboolean('Parameters','plotLinRegression')
            self.addSampleTagPlot = self.conf.getboolean('Parameters','addSampleTagPlot')
            self.polyDegree = self.conf.getint('Parameters','polyDegree')
            self.plotSpecificColors = self.conf.getboolean('Parameters','plotSpecificColors')
            self.columnSpecColors = self.conf.getint('Parameters','columnSpecColors')
            self.customColors = self.conf.getboolean('Parameters','customColors')
            self.custColorsList = ast.literal_eval(self.corrAnalysisPar['custColorsList'])

            self.graphX = ast.literal_eval(self.corrAnalysisPar['graphX'])
            self.graphY = ast.literal_eval(self.corrAnalysisPar['graphY'])
    
            self.plotSpectralCorr = self.conf.getboolean('Parameters','plotSpectralCorr')                # True: use for raw data (spectra, etc)
            self.stepXticksPlot = self.conf.getint('Parameters','stepXticksPlot')
            self.corrSpectraFull = self.conf.getboolean('Parameters','corrSpectraFull')
            self.corrSpectraMin = self.conf.getfloat('Parameters','corrSpectraMin')
            
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.corrAnalysisDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 CorrAnalysis <paramFile>')
        print(' Usage:\n  python3 CorrAnalysis <paramFile> <validFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dP = Conf()
        
    dfP,dfL = readParamFile(sys.argv[1], dP)
    if dfP is None:
        print("Exiting due to error reading parameter file.")
        return 0
        
    if dP.separateValidFile and len(sys.argv) > 2:
        dfV, _ = readParamFile(sys.argv[2], dP)
        #dfP = dfP.append(dfV,ignore_index=True) # deprecated in Pandas v>2
        dfP = pd.concat([dfP, dfV], ignore_index=True)
        dP.validRows = dfP.index.tolist()[-len(dfV.index.tolist()):]
    P,headP = processParamFile(dfP, dP.trainCol, dP)
    V,headV = processParamFile(dfP, dP.predCol, dP)
    
    #rootFile = os.path.splitext(sys.argv[1])[0]
    rootFile = os.path.splitext((os.path.basename(sys.argv[1])))[0]
    pearsonFile = rootFile + '_pearsonR.csv'
    spearmanFile = rootFile + '_spearmanR.csv'
    plotFile = rootFile + '_plots.pdf'
    spearmanSummary = rootFile + '_spearmanR_summary' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    pearsonSummary = rootFile + '_pearsonR_summary' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))

    pearsonR, spearmanR = getCorrelations (V, P, dP)
    
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

    #corr = dfP.corr(method='pearson')
    #heatMapsCorrelations2(dfP)

    if dP.heatMapsCorr:
        print(" Correlation heat maps saved in:",plotFile,"\n")
        heatMapsCorrelations(dfPearson, "PearsonR_correlation", pdf, dP)
        heatMapsCorrelations(dfSpearman, "SpearmanR_correlation", pdf,dP)
    if dP.plotSpectralCorr:
        print(" Correlation plots saved in:",plotFile,"\n")
        plotSpectralCorrelations(dfPearson, P, "PearsonR_correlation", rootFile, pdf, dP)
        plotSpectralCorrelations(dfSpearman, P, "SpearmanR_correlation", rootFile, pdf, dP)
        if not dP.corrSpectraFull:
            dfPearson_tmp = dfPearson.copy()
            dfSpearman_tmp = dfSpearman.copy()
            dfPearson_tmp[dfPearson_tmp < dP.corrSpectraMin] = 0
            dfSpearman_tmp[dfSpearman_tmp < dP.corrSpectraMin] = 0
            plotSpectralCorrelations(dfPearson_tmp, P, "PearsonR_correlation (Corr > "+ str(dP.corrSpectraMin)+")", rootFile, pdf)
            plotSpectralCorrelations(dfSpearman_tmp, P, "SpearmanR_correlation", rootFile, pdf)

    if dP.plotSelectedGraphs:
        num1 = plotSelectedGraphs(dfP, dfL, dfPearson, dP.graphX, dP.graphY, dP.validRows, "PearsonR_correlation", pdf, dP)
        num2 = plotSelectedGraphs(dfP, dfL, dfSpearman, dP.graphX, dP.graphY, dP.validRows, "SpearmanR_correlation", pdf, dP)
        print(" ",num1+num2,"Manually selected plots saved in:",plotFile,"\n")

    if dP.plotGraphsThreshold:
        num1 = plotGraphThreshold(dfP, dfL, dfPearson, dP.validRows, "PearsonR_correlation", pdf, pearsonSummary, dP)
        num2 = plotGraphThreshold(dfP, dfL, dfSpearman, dP.validRows, "SpearmanR_correlation", pdf, spearmanSummary, dP)
        print(" ",num1+num2,"XY plots with correlation in [",dP.corrMin,",",dP.corrMax,"] saved in:",plotFile,"\n")
    pdf.close()
    
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, dP):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",", skiprows=dP.skipHeadRows)
        print(dfP)
        if dP.skipHeadRows != 0 :
            with open(paramFile, 'r') as f:
                dfL = pd.read_csv(f, delimiter = ",", nrows=dP.skipHeadRows)
            dfL.columns = dfL.iloc[dP.skipHeadRows-1]
            print(dfL.columns)
        else:
            dfL = pd.DataFrame([])
    except:
        print("\033[1m Param file:",paramFile,"not found/broken. \n\033[0m")
        sys.exit()
    return dfP, dfL

def processParamFile(dfP, lims, dP):
    if lims[0]>len(dfP.columns):
        lims[0] = len(dfP.columns)
        print(" Warning: Column range is larger than actual number of columns. Using full dataset")
    
    P = dfP.iloc[:,lims]
    
    if dP.skipEmptyColumns:
        # Use this for only empty columns
        #cols_to_drop = P.columns[(P == 0).all()]
        
        # Remove constant columns
        cols_to_drop = []
        for col in P.columns:
            if (P[col] == P[col].iloc[0]).all():
                cols_to_drop.append(col)
                
        P = P.drop(cols_to_drop, axis=1)
    
    headP = P.columns.values
    P = P.to_numpy()
    
    P[np.isnan(P)] = dP.valueForNan
    return P, headP
    
#************************************
# Calculate Correlations
#************************************
def getCorrelations(V, P, dP):
    pearsonR=np.empty((V.shape[1],P.shape[1]))
    spearmanR=np.empty((V.shape[1],P.shape[1]))
    for j in range(V.shape[1]):
        for i in range(P.shape[1]):
            P2, V2, _, indx = purgeSparse(P[:,i], V[:,j], P[:,i], dP)
            try:
                # Check size explicitly before calling, as correlation on < 2 points is undefined
                if P2.size < 2 or V2.size < 2:
                    pearsonR[j,i] = np.nan
                    spearmanR[j,i] = np.nan
                else:
                    pearsonR[j,i], _ = pearsonr(P2, V2)
                    spearmanR[j,i], _ = spearmanr(P2, V2)
            except ValueError: # Catch error just in case size check isn't exhaustive
                pearsonR[j,i] = np.nan
                spearmanR[j,i] = np.nan

    return pearsonR, spearmanR

def purgeSparse(P, V, label, dP):
    if dP.removeNaNfromCorr:
        pt = []
        vt = []
        ann = []
        ind = []
        for l in range (P.shape[0]):
            # EXPERIMENTAL: Use this to remove additional points.
            #if P[l] != dP.valueForNan and V[l] != dP.valueForNan and V[l] > 1:
            # USE THIS FOR REGULAR USE
            if P[l] != dP.valueForNan and V[l] != dP.valueForNan:
                pt.append(P[l])
                vt.append(V[l])
                ann.append(label[l])
                ind.append(l)
        P2=np.array(pt)
        V2=np.array(vt)
    else:
        P2 = P
        V2 = V
        ann = label
        ind = range (P.shape[0])
        
    if P2.size < 2 or V2.size <2:
        P2 = np.array([0,0])
        V2 = np.array([0,0])
    return P2, V2, ann, ind

def getCorrelationsExperimental(dfP):
    dfPearson = dfP.corr(method='pearson')
    dfSpearman = dfP.corr(method='spearman')

#************************************
# Plot Heat Maps Correlations
#************************************
def heatMapsCorrelations(dfP, title, pdf, dP):
    data = dfP.to_numpy()
    Rlabels = dfP.index.tolist()
    Clabels = dfP.columns.values
    fig, ax = plt.subplots(figsize=(20, 8))
    cmap = mpl.colormaps['viridis']

    if not dP.heatMapCorrFull:
        if dP.corrMax > 0:
            data = np.where(data < dP.corrMin, dP.corrMin, data)
        else:
            data = np.where(data > dP.corrMax, dP.corrMax, data)
            cmap = cmap.reversed()
    
    im = ax.imshow(data, cmap = cmap)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("correlation", rotation=-90, va="bottom")
    
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
    
    ax.set_title(title)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    pdf.savefig()
    #plt.savefig(title+".png", dpi = 160, format = 'png')  # Save plot
    #plt.show()
    plt.close()

#*********************************************************************
# Create custom palette based on values from a column in the dataset
#*********************************************************************
def createPalette(dP, dfP, indx):
    sampleColors = dfP.iloc[:,dP.columnSpecColors+dP.skipHeadColumns].tolist()
    purgedSampleColors = [sampleColors[i]-1 for i in indx]
    if dP.customColors:
        cmap = colors.ListedColormap(dP.custColorsList)
        mapped = [cmap(val) for val in purgedSampleColors]
    else:
        mapped=cm.rainbow(np.array(purgedSampleColors)/np.mean(purgedSampleColors))
    return purgedSampleColors, mapped

#************************************
# Plot Graphs based on threshold
#************************************
def plotGraphThreshold(dfP, dfL, dfC, validRows, title, pdf, sumFile, dP):
    num = 0
    dfSummary = pd.DataFrame()
    for col in dfC.columns:
        for ind in dfC[dfC[col].between(dP.corrMin,dP.corrMax)].index:
            x, y, ann, indx = purgeSparse(dfP[col].to_numpy(), dfP[ind].to_numpy(), dfP.iloc[:,0], dP)
            if dP.plotSpecificColors:
                if dP.customColors:
                    color, mapped = createPalette(dP, dfP, indx)
                    plt.scatter(x, y, marker='o', c=mapped)
                else:
                    mapped = np.random.randint(0, len(dfP[dfL.columns[dP.columnSpecColors]].unique().tolist()), len(x))
                    scatter = plt.scatter(x, y, marker='o', c=mapped, cmap=dP.cmap)
                    handles, labels = scatter.legend_elements()
                    plt.legend(handles=handles, labels=labels, title=dfL.columns[dP.columnSpecColors])
            else:
                plt.plot(x,y, 'bo')
                
            if dfL.empty:
                xlabel = col
                ylabel = ind
            else:
                xlabel = formatLabels(dfL, col)
                ylabel = formatLabels(dfL, ind)
               
            dfSummary = pd.concat([dfSummary, pd.DataFrame([{'PAR': xlabel, 'PERF': ylabel, 'Corr': dfC[col].loc[ind], 'Num_points': len(x), 'Valid': 'NO'}])], ignore_index=True)
        
            if dP.plotValidData:
                print("\nValid datapoint:\n",dfP.loc[validRows,col])
                xv, yv, ann, indx = purgeSparse(dfP.loc[validRows,col].to_numpy(), dfP.loc[validRows, ind].to_numpy(), dfP.iloc[:,0], dP)
                dfSummary = pd.concat([dfSummary, pd.DataFrame([{'PAR': xlabel, 'PERF': ylabel, 'Corr': dfC[col].loc[ind], 'Num_points': len(xv), 'Valid': 'YES'}])], ignore_index=True)
                if dP.plotSpecificColors:
                    if dP.customColors:
                        color, mapped = createPalette(dP, dfP, indx)
                        plt.scatter(xv, yv, marker='x', c=mapped)
                    else:
                        mapped = np.random.randint(0, len(dfP[dfL.columns[dP.columnSpecColors]].unique().tolist()), len(x))
                        scatter = plt.scatter(xv, yv, marker='x', c=mapped, cmap=dP.cmap)
                        handles, labels = scatter.legend_elements()
                        plt.legend(handles=handles, labels=labels, title=dfL.columns[dP.columnSpecColors])
                else:
                    plt.plot(xv, yv, 'ro')
                
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title+": {0:.3f}".format(dfC[col].loc[ind]))
            
            if dP.addSampleTagPlot:
                for k, txt, in enumerate(ann):
                    plt.annotate(txt,xy=(x[k],y[k]), fontsize='x-small')
            if dP.plotLinRegression and col is not ind:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    plt.text(min(x), max(y),"{0:s} = {2:.3f}*{1:s} + {3:.3f}".format(ind, col, slope, intercept))
                    #plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, dP.polyDegree))(np.unique(x)))
                    plt.plot(np.unique(x), polyfit.fit(x, y, dP.polyDegree)(np.unique(x)))
                except:
                    pass
            #plt.legend(loc='upper left')
            pdf.savefig()
            plt.close()
            num+=1
    if num ==0:
        print(" All correlations are outside the requested range\n")
    else:
        dfSummary = dfSummary.sort_values(by=['PERF'])
        print(title, "\n", dfSummary, "\n")
        dfSummary.to_csv(sumFile, index=True, header=True)
    return num
    
#************************************
# Plot Graphs based on manual input
#************************************
def plotSelectedGraphs(dfP, dfL, dfC, X, Y, validRows, title, pdf, dP):
    num = 0
    for i in X:
        for j in Y:
            if dfL.empty:
                xlabel = dfP.columns.values[i]
                ylabel = dfP.columns.values[j]
            else:
                xlabel = formatLabels(dfL, dfP.columns.values[i])
                ylabel = formatLabels(dfL, dfP.columns.values[j])
            
            P, V, ann, indx = purgeSparse(dfP.iloc[:,i].to_numpy(), dfP.iloc[:,j].to_numpy(), dfP.iloc[:,0], dP)
            if dP.plotSpecificColors:
                if dP.customColors:
                    color, mapped = createPalette(dP, dfP, indx)
                    plt.scatter(P, V, marker='o', c=mapped)
                else:
                    mapped = np.random.randint(0, len(dfP[dfL.columns[dP.columnSpecColors]].unique().tolist()), len(x))
                    scatter = plt.scatter(P, V, marker='o', c=mapped, cmap=dP.cmap)
                    handles, labels = scatter.legend_elements()
                    plt.legend(handles=handles, labels=labels, title=dfL.columns[dP.columnSpecColors])
            else:
                plt.plot(P,V, 'bo')
            
            if dP.plotValidData:
                PV, VV, ann, indx = purgeSparse(dfP.iloc[validRows,i].to_numpy(), dfP.iloc[validRows, j].to_numpy(), dfP.iloc[validRows, 0], dP)
                if dP.plotSpecificColors:
                    if dP.customColors:
                        color, mapped = createPalette(dP, dfP, indx)
                        plt.scatter(PP, VV, marker='x', c=mapped)
                    else:
                        mapped = np.random.randint(0, len(dfP[dfL.columns[dP.columnSpecColors]].unique().tolist()), len(x))
                        scatter = plt.scatter(PP, VV, marker='x', c=mapped, cmap=dP.cmap)
                        handles, labels = scatter.legend_elements()
                        plt.legend(handles=handles, labels=labels, title=dfL.columns[dP.columnSpecColors])
                else:
                    plt.plot(PV,VV, 'ro')
                
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title+": {0:.3f}".format(dfC[dfP.columns.values[i]].loc[dfP.columns.values[j]]))
            
            if dP.addSampleTagPlot:
                for k, txt, in enumerate(ann):
                    plt.annotate(txt,xy=(P[k],V[k]), fontsize='x-small')
            #plt.legend(loc='upper left')
            
            if dP.plotLinRegression and dfP.columns.values[i] is not dfP.columns.values[j]:
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(P,V)
                    plt.text(min(P), max(V),"{0:s} = {2:.3f}*{1:s} + {3:.3f}".format(dfP.columns.values[j], dfP.columns.values[i], slope, intercept))
                    #plt.plot(np.unique(P), np.poly1d(np.polyfit(P, V, dP.polyDegree))(np.unique(P)))
                    plt.plot(np.unique(P), polyfit.fit(P, V, dP.polyDegree)(np.unique(P)))
                except:
                    pass
            
            pdf.savefig()
            plt.close()
            num += 1
    return num

#************************************
# Plot Spectral Correlations
#************************************
def plotSpectralCorrelations(dfP, P, title, filename, pdf, dP):
    data = dfP.to_numpy()
    
    plt.xlabel('Wavelength')
    plt.ylabel('Correlation')
    
    Rlabels = dfP.index.tolist()
    Clabels = np.float_(dfP.columns.values)
    Clabels_plot = Clabels[::dP.stepXticksPlot]
    
    fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize=(10, 10))
    
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
    plt.close()

#************************************
# Format Labels
#************************************
def formatLabels(dfL, ind):
    label = ""
    for l in range(len(dfL)):
        if str(dfL[ind][l]) != "nan":
            label += " " + str(dfL[ind][l])
    return label.replace('\n',' ')
    
#*************************************************************
# Check and identify where are badly typed entries in dataset
#*************************************************************
def checkBadTypesDataset(P):
    def is_not_numeric(x):
        return not isinstance(x, (int, float))
        
    vectorized_checker = np.vectorize(is_not_numeric)
    problematic_indices = np.where(vectorized_checker(P))

    if P.dtype == 'object':
        print(f"\n Type issue: {P.dtype}")
        print(f" Problematic entries are at indices: {problematic_indices}")
        print(f" The problematic values are: {P[problematic_indices]}\n")
        return False
    else:
        print("\n No type issues detected.\n")
        return True

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
