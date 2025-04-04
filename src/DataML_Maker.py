#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_Maker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: 2025.04.04.2
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, h5py, pickle, configparser
from random import uniform
from bisect import bisect_left
from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_Maker():
    main()

#************************************
# Parameters definition
#************************************
class Conf():
    
    def __init__(self):
        self.appName = "DataML_Maker"
        confFileName = "DataML_Maker.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "./"
        
        # Do not change
        def rescaleList(list, value):
            list = [x + value for x in list]
            return list
        
        if self.fullDataset:
            self.minCCol = self.minCCol + self.numHeadColumns-1
            self.maxCCol = self.maxCCol + self.numHeadColumns-1
        else:
            self.charCCols = rescaleList(self.charCCols, self.numHeadColumns - 1)
        
        self.predRColTag = self.predRCol
        self.predRCol = rescaleList(self.predRCol, self.numHeadColumns - 1)
        
        self.numLabels = len(self.predRCol)
        self.validRows = [x-1 for x in self.validRows]
    
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
            self.dataMLMakerDef = self.conf['Parameters']

            self.saveAsTxt = self.conf.getboolean('Parameters','saveAsTxt')
            self.numHeadColumns = self.conf.getint('Parameters','numHeadColumns')
            self.numHeadRows = self.conf.getint('Parameters','numHeadRows')
            self.fullDataset = self.conf.getboolean('Parameters','fullDataset')
            
            self.minCCol = self.conf.getint('Parameters','minCCol')
            self.maxCCol = self.conf.getint('Parameters','maxCCol')
            self.charCCols = eval(self.dataMLMakerDef['charCCols'])
            self.predRCol = eval(self.dataMLMakerDef['predRCol'])
            
            self.purgeUndefRows = self.conf.getboolean('Parameters','purgeUndefRows')
            self.validFile = self.conf.getboolean('Parameters','validFile')
            self.createRandomValidSet = self.conf.getboolean('Parameters','createRandomValidSet')
            
            self.percentValid = self.conf.getfloat('Parameters','percentValid')
            self.validRows = eval(self.dataMLMakerDef['validRows'])
            self.precData = self.conf.getint('Parameters','precData')
            
            self.saveNormalized = self.conf.getboolean('Parameters','saveNormalized')            #
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.useCustomRound = self.conf.getboolean('Parameters','useCustomRound')
            
            self.YnormTo = self.conf.getfloat('Parameters','YnormTo')
            self.stepNormLabel = self.conf.getfloat('Parameters','stepNormLabel')
            self.randomize = self.conf.getboolean('Parameters','randomize')
            self.fullRandomMatrix = self.conf.getboolean('Parameters','fullRandomMatrix')
            
            self.numRandomAdds = self.conf.getint('Parameters','numRandomAdds')
            self.randomCols = eval(self.dataMLMakerDef['randomCols'])
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
    

#************************************
# Main
#************************************
def main():
    dP = Conf()
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 DataMaker.py <paramFile> <pred column - optional>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    if len(sys.argv) >= 3:
        predRCol = [int(sys.argv[2])]
        predRColTag = predRCol
    else:
        predRCol = dP.predRCol
        predRColTag = dP.predRColTag
    
    if dP.fullDataset:
        datasetLabel = '_fullDataSet'
    else:
        datasetLabel = '_partialDataSet'
    
    rootFile = os.path.splitext((os.path.basename(sys.argv[1])))[0] + datasetLabel
    
    if dP.purgeUndefRows:
        rootFile += '_purged'
        
    rootFile += '_p' + str(predRColTag[0])
    learnFile = rootFile + '_train'
    
    try:
        P,V = readParamFile(sys.argv[1], predRCol, rootFile, dP)
    except:
        print("\033[1m" + " Something went wrong, maybe Param file not found\n" + "\033[0m")
        return
    
    #************************************
    # Creating training set
    #************************************
    if dP.randomize:
        print(" Creating randomized training set using",dP.minPercVariation*100, "% as max variation on parameters\n")
        Pr = randomize(P)
        
        if dP.fullRandomMatrix:
            randTag = '_fullrand'
        else:
            randTag = '_partrand'
       
        learnRandomFile = learnFile + randTag + str(dP.numRandomAdds) + '_pcVar' + str(int(dP.minPercVariation*100))
        
        if dP.randomizeLabel:
            learnRandomFile += '_rLab'
        if dP.saveNormalized or dP.normalizeLabel:
            norm = Normalizer(Pr, dP)
            norm.save(rootFile +"_norm.pkl")
            saveLearnFile(norm.transform_matrix(Pr), learnRandomFile+'_norm', True, dP)
        else:
            saveLearnFile(Pr, learnRandomFile, False, dP)
    else:
        if dP.saveNormalized or dP.normalizeLabel:
            norm = Normalizer(P, dP)
            norm.save(rootFile+ "_norm.pkl")
            saveLearnFile(norm.transform_matrix(P), learnFile +'_norm', True, dP)
        else:
            saveLearnFile(P, learnFile, False, dP)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, predRCol, rootFile, dP):
    if dP.fullDataset:
        usecols = range(dP.minCCol,dP.maxCCol)
    else:
        usecols = dP.charCCols
    
    with open(paramFile, 'r') as f:
        P2 = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows).to_numpy()
    
    M = np.hstack((P2[:,predRCol],P2[:,usecols]))

    if dP.purgeUndefRows:
        M = purgeRows(M)
    
    P = np.vstack([list(range(0,M.shape[1])),M])
    V = np.array([list(range(0,M.shape[1]))])

    #***************************************
    # Handle Validation File
    #***************************************
    if dP.validFile:
        validFile = rootFile + '_test'
        
        if dP.createRandomValidSet:
            P, V = formatSubset(P, dP.percentValid)
        else:
            if dP.validRows != 0:
                P = np.vstack([list(range(0,M.shape[1])),np.delete(M,dP.validRows,0)])
                V = np.vstack([list(range(0,M.shape[1])),M[dP.validRows,:]])
                
        if dP.saveNormalized or dP.normalizeLabel:
            saveLearnFile(norm.transform_matrix(V), validFile +'_norm', True, dP)
        else:
            saveLearnFile(V, validFile, False, dP)
    return P,V
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile, saveNormFlag, dP):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'ab') as f:
                 np.savetxt(f, M, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        if saveNormFlag == False:
            print(" Saving new training file (txt) in:", learnFile+"\n")
        else:
            print(" Saving new normalized training file (txt) in:", learnFile+"\n")
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M.astype(np.float64))

#***************************************
# Randomize initial set
#***************************************
def randomize(P):
    Pr = np.copy(P)

    if not dP.fullRandomMatrix:
        cols = dP.randomCols
    else:
        cols = P[0,1:].astype(int)
        
    for j in range (0,dP.numRandomAdds):
        rand = randomMatrix(P, cols)
        temp = np.multiply(P[1:],rand)
        Pr = np.vstack([Pr, temp])
    return Pr

def randomMatrix(P, cols):
    rand = np.ones(P[1:].shape)
    rand[:,cols] = np.random.uniform(1-dP.minPercVariation,1,rand[:,cols].shape)
    return rand
    
#************************************
# Create validation subset
#************************************
def formatSubset(A, percent):
    from sklearn.model_selection import train_test_split
    
    numValid = round(A[1:,:].shape[0]*percent)
    numTrain = round(A[1:,:].shape[0] - numValid)
    print(" Creating a training set with:", str(numTrain), "datapoints")
    print(" Creating a validation set with:", str(numValid), "datapoints\n")
    
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A[1:,1:], A[1:,0], test_size=percent, random_state=42)
    Atrain = np.vstack((A[0,:],np.hstack((Cl_train.reshape(1,-1).T, A_train))))
    Atest = np.vstack((A[0,:],np.hstack((Cl_cv.reshape(1,-1).T, A_cv))))
    
    return Atrain, Atest
'''
def formatSubset2(A, Cl, percent):
    list = np.random.choice(range(len(Cl)), int(np.rint(percent*len(Cl))), replace=False)
    A_train = np.delete(A,list,0)
    Cl_train = np.delete(Cl,list)
    A_cv = A[list]
    Cl_cv = Cl[list]
    return A_train, Cl_train, A_cv, Cl_cv
'''

#************************************
# Purge rows with undefined Cl value
#************************************
def purgeRows(M):
    condition = M[:,0] == 0
    M2 = M[~condition]
    print(" Shape original dataset:", M.shape)
    print(" Purged from undefined values. \n Shape new dataset:",M2.shape,"\n")
    return M2

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
