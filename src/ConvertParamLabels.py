#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* ConvertParamLabels
* Convert progressive numeric labels with actual parameter names
* Uses DataML_Maker.ini
* version: 2016.04.10.2
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
            print(f" Configuration file: {confFileName} not found. Aborting.")
            print(f" You should use {confFileName} from the initial training file generation.\n")
            sys.exit(1)
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        self.fullDataset = True
        self.minCCol = self.minCCol + self.numHeadColumns-1
        self.maxCCol = self.maxCCol + self.numHeadColumns-1
            
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.dataMLMakerPar = self.conf['Parameters']
            
            self.numHeadColumns = self.conf.getint('Parameters','numHeadColumns')
            self.numHeadRows = self.conf.getint('Parameters','numHeadRows')
            self.minCCol = self.conf.getint('Parameters','minCCol')
            self.maxCCol = self.conf.getint('Parameters','maxCCol')
        
        except:
            print(" Error in reading configuration file. Please check it\n")

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
    
    print(" config.txt:",configData)
    print(" Converted labels from config.txt",label_list)
    
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
