#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CheckDataSet
* Uses CorrAnalysis.ini
* version: 2025.08.21.1
* By: Nicola Ferralis <feranick@hotmail.com>
* Licence: GPL 2 or newer
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, configparser, ast

#***************************************************
# This is needed for installation through pip
#***************************************************
def CheckDataSet():
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
        print(' Usage:\n  python3 CheckDataSet <paramFile>')
        print(' Usage:\n  python3 CheckDataSet <paramFile> <validFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    dP = Conf()
        
    dfP = readParamFile(sys.argv[1], dP)
    if dfP is None:
        print("Exiting due to error reading parameter file.")
        return 0
        
    if dP.separateValidFile and len(sys.argv) > 2:
        dfV, _ = readParamFile(sys.argv[2], dP)
        #dfP = dfP.append(dfV,ignore_index=True) # deprecated in Pandas v>2
        dfP = pd.concat([dfP, dfV], ignore_index=True)
        dP.validRows = dfP.index.tolist()[-len(dfV.index.tolist()):]
        
    checkBadTypesDataset_new(dfP)
        
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, dP):
    try:
        with open(paramFile, 'r') as f:
            dfP = pd.read_csv(f, delimiter = ",",skiprows=dP.skipHeadRows)
    except:
        print("\033[1m Param file:",paramFile," not found/broken \n\033[0m")
        return
    return dfP
    
#*************************************************************
# Check and identify where are badly typed entries in dataset
#*************************************************************
def checkBadTypesDataset(dfP):
    non_numeric_cols = dfP.select_dtypes(exclude=np.number).columns
    #non_numeric_cols = dfP.select_dtypes(include="object").columns
    non_numeric_row = dfP[~dfP[non_numeric_cols[1]].str.isnumeric()].index.tolist()
    non_numeric_strings = dfP[~dfP[non_numeric_cols[1]].str.isnumeric()][non_numeric_cols[1]].to_numpy()
    print(len(non_numeric_cols), len(non_numeric_row), len(non_numeric_strings))
    print(non_numeric_cols, non_numeric_row, non_numeric_strings)
    for i in range(len(non_numeric_row)):
        print(f"\nNon-numeric strings in \n column: {non_numeric_cols[i+1]} \n row: {non_numeric_row[i]} \n value: {non_numeric_strings[i]}\n")
        
def checkBadTypesDataset_new(dfP):
    non_numeric_cols = dfP.select_dtypes(exclude=np.number).columns
    print(f"\nNon-numeric strings in:\n")
    for i in range(1,len(non_numeric_cols),1):
        print(i,non_numeric_cols[i])
        non_numeric_row = dfP[~dfP[non_numeric_cols[i]].str.isnumeric()].index.tolist()
        non_numeric_strings = dfP[~dfP[non_numeric_cols[i]].str.isnumeric()][non_numeric_cols[i]].to_numpy()
        print(non_numeric_row)
        print(non_numeric_strings)
        #for j in range(len(non_numeric_row)):
        #    print(f" column: {non_numeric_cols[i]} \n row: {non_numeric_row[j]} \n value: {non_numeric_strings[j]}\n")

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
