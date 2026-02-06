#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_BatchMaker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: 2026.02.06.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py, pickle, configparser, ast
from random import uniform
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_BatchMaker():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    
    def __init__(self):
        self.appName = "DataML_BatchMaker"
        confFileName = "DataML_BatchMaker.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        if self.fullDataset:
            dP.useCols = range(dP.minCCol,dP.maxCCol)
            self.minCCol = self.minCCol + self.numHeadColumns-1
            self.maxCCol = self.maxCCol + self.numHeadColumns-1
        else:
            self.useCols = self.charCCols
            self.charCCols = self.rescaleList(self.charCCols, self.numHeadColumns - 1)
        
        #self.numLabels = len(self.predRCol)
        self.validRows = [x-1 for x in self.validRows]
    
    def dataMLMakerDef(self):
        self.conf['Parameters'] = {
            'numHeadColumns' : 2,
            'numHeadRows' : 0,
            'fullDataset' : False,
            'minCCol' : 1,
            'maxCCol' : 42,
            'charCCols' : [21,23,25,34],
            'sampleLabelCol' : 0,
            'validRows' : [1,2,3],
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.dataMLMakerPar = self.conf['Parameters']

            self.numHeadColumns = self.conf.getint('Parameters','numHeadColumns')
            self.numHeadRows = self.conf.getint('Parameters','numHeadRows')
            self.fullDataset = self.conf.getboolean('Parameters','fullDataset')
            
            self.minCCol = self.conf.getint('Parameters','minCCol')
            self.maxCCol = self.conf.getint('Parameters','maxCCol')
            self.charCCols = ast.literal_eval(self.dataMLMakerPar['charCCols'])
            self.sampleLabelCol = self.conf.getint('Parameters','sampleLabelCol')
            self.validRows = ast.literal_eval(self.dataMLMakerPar['validRows'])
            
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.dataMLMakerDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
                
        except Exception as e:
            print("Error in creating configuration file:")
            print(f"  {e}\n")
            
    # Do not change
    def rescaleList(self, list, value):
        list = [x + value for x in list]
        return list
    
#************************************
# Main
#************************************
def main():
    dP = Conf()
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 DataML_BatchMaker.py <paramFile>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    batchFile = os.path.splitext((os.path.basename(sys.argv[1])))[0] + "_batch"
    
    try:
        makeBatchFile(sys.argv[1], batchFile, dP)
    except Exception as e:
        print(f" An error occurred: {e}\n")
        return
    

#************************************
# Make Batch File
#************************************
def makeBatchFile(paramFile, batchFile, dP):
    
    print(f" Opening parameter file: {paramFile}\n")
    with open(paramFile, 'r') as f:
        df_full = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows)
    
    df = df_full.iloc[dP.validRows]
    
    paramNames = df.columns[dP.charCCols].tolist()
    sampleNames = df[df.columns[dP.sampleLabelCol]].values
    
    batchFile += "_"+str(len(paramNames))+"params.csv"
    
    print(f" Parameter Names: {paramNames}\n")
    print(f" Sample Names: {sampleNames}\n")
    
    df2 = df[df.columns[dP.charCCols]].T
    df2.columns = sampleNames
    
    print(f" New Batch Matrix:\n {df2}\n")
    
    df2.to_csv(batchFile)
    
    print(f" New Batch File: {batchFile}\n")
    
    return True

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
