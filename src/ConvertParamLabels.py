#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* ConvertParamLabels
* Convert progressive numeric labels with actual parameter names
* Uses DataML_Datamaker.ini
* version: 2025.12.19.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import pandas as pd
import os, sys, os.path, h5py, pickle, configparser, ast, csv
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def ConvertParamLabels():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    
    def __init__(self):
        self.appName = "DataML_Maker"
        confFileName = "DataML_Maker.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        self.fullDataset = True
        self.minCCol = self.minCCol + self.numHeadColumns-1
        self.maxCCol = self.maxCCol + self.numHeadColumns-1
        
    
    def dataMLMakerDef(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'numHeadColumns' : 2,
            'numHeadRows' : 0,
            'fullDataset' : False,
            'minCCol' : 1,
            'maxCCol' : 42,
            'charCCols' : [21,23,25,34],
            'predRCol' : [43],
            'purgeUndefRows' : False,
            'validFile' : True,
            'createRandomValidSet' : False,
            'percentValid' : 0.05,
            'validRows' : [1,2,3],
            'precData' : 3,
            'saveNormalized' : False,
            'normalizeLabel' : False,
            'useCustomRound' : True,
            'YnormTo' : 1,
            'stepNormLabel' : 0.001,
            'randomize' : False,
            'fullRandomMatrix' : True,
            'numRandomAdds' : 50,
            'randomCols' : [3],
            'minPercVariation' : 0.05,
            'randomizeLabel' : False,
            'useGeneralNormLabel' : False,
            'minGeneralLabel' : 10,
            'maxGeneralLabel' : 60,
            'valueForNan' : -1,
            }
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.dataMLMakerPar = self.conf['Parameters']

            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.numHeadColumns = self.conf.getint('Parameters','numHeadColumns')
            self.numHeadRows = self.conf.getint('Parameters','numHeadRows')
            self.fullDataset = self.conf.getboolean('Parameters','fullDataset')
            
            self.minCCol = self.conf.getint('Parameters','minCCol')
            self.maxCCol = self.conf.getint('Parameters','maxCCol')
            self.charCCols = ast.literal_eval(self.dataMLMakerPar['charCCols'])
            self.predRCol = ast.literal_eval(self.dataMLMakerPar['predRCol'])
            
            self.purgeUndefRows = self.conf.getboolean('Parameters','purgeUndefRows')
            self.validFile = self.conf.getboolean('Parameters','validFile')
            self.createRandomValidSet = self.conf.getboolean('Parameters','createRandomValidSet')
            
            self.percentValid = self.conf.getfloat('Parameters','percentValid')
            self.validRows = ast.literal_eval(self.dataMLMakerPar['validRows'])
            self.precData = self.conf.getint('Parameters','precData')
            
            self.saveNormalized = self.conf.getboolean('Parameters','saveNormalized')            #
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.useCustomRound = self.conf.getboolean('Parameters','useCustomRound')
            
            self.YnormTo = self.conf.getfloat('Parameters','YnormTo')
            self.stepNormLabel = self.conf.getfloat('Parameters','stepNormLabel')
            self.randomize = self.conf.getboolean('Parameters','randomize')
            self.fullRandomMatrix = self.conf.getboolean('Parameters','fullRandomMatrix')
            
            self.numRandomAdds = self.conf.getint('Parameters','numRandomAdds')
            self.randomCols = ast.literal_eval(self.dataMLMakerPar['randomCols'])
            self.minPercVariation = self.conf.getfloat('Parameters','minPercVariation')
            self.randomizeLabel = self.conf.getboolean('Parameters','randomizeLabel')
            self.useGeneralNormLabel = self.conf.getboolean('Parameters','useGeneralNormLabel')
            self.minGeneralLabel = self.conf.getint('Parameters','minGeneralLabel')
            self.maxGeneralLabel = self.conf.getint('Parameters','maxGeneralLabel')
            self.valueForNan = self.conf.getfloat('Parameters','valueForNan')
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.dataMLMakerDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")
            
    # Do not change
    def rescaleList(self, list, value):
        list = [x + value for x in list]
        return list
    
#************************************
# Main
#************************************
def main():
    dP = Conf()
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 ConvertParamLabels.py <paramFile> <config.txt>')
        print('\n  Requires DataML_Maker.ini to understand the data structure of the <paramFile>')
        print('  Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    rootFile = os.path.splitext((os.path.basename(sys.argv[2])))[0]
    oldConfigFile = rootFile+"_numeric.txt"
    newConfigFile = sys.argv[2]
    
    try:
        featNames = readParamFile(sys.argv[1], dP)
        configData = readConfigFile(sys.argv[2])
    except Exception as e:
        print(f" An error occurred: {e}\n")
        return
    
    label_list = convertNumLabel(featNames,configData)
    
    print(" Config.ini:",configData)
    print(" Converted labels from onfig.ini",label_list)
    
    os.rename(sys.argv[2], oldConfigFile)
    print("\n Old",sys.argv[2],"renamed:", oldConfigFile)
    
    saveConfFile(newConfigFile, label_list)
    print(" New converted config file:", newConfigFile,"\n")
        
#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, dP):
    usecols = range(dP.minCCol,dP.maxCCol)
    
    with open(paramFile, 'r') as f:
        df = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows)
    featNames = df.columns.to_list()[dP.numHeadColumns:]
    
    return featNames
    
def readConfigFile(configFile):
    data_list = []
    with open(configFile, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            data_list.append(row)
    converted_list = [int(item[1:])-1 for item in data_list[0]]
    return converted_list
    
def convertNumLabel(featNames,configData):
    label_list = []
    for i in configData:
        label_list.append(featNames[i])
    return label_list
    
#***************************************
# Save new learning Data
#***************************************
def saveConfFile(csvFile, label_list):
    with open(csvFile, 'w', newline='') as f:
        # Create a csv.writer object
        csv_writer = csv.writer(f)
        csv_writer.writerow(label_list)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
