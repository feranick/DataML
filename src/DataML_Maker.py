#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_Maker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: 2026.02.26.2
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
def DataML_Maker():
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
        
        if self.fullDataset:
            self.minCCol = self.minCCol + self.numHeadColumns-1
            self.maxCCol = self.maxCCol + self.numHeadColumns-1
        else:
            self.charCCols = self.rescaleList(self.charCCols, self.numHeadColumns - 1)
        
        self.predRColTag = self.predRCol
        self.predRCol = self.rescaleList(self.predRCol, self.numHeadColumns - 1)
        
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
            'numGroupCols' : 5,
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
            'convertToNAN' : False,
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
            self.numGroupCols = self.conf.getint('Parameters','numGroupCols')
            
            self.percentValid = self.conf.getfloat('Parameters','percentValid')
            self.validRows = ast.literal_eval(self.dataMLMakerPar['validRows'])
            self.precData = self.conf.getint('Parameters','precData')
            
            self.saveNormalized = self.conf.getboolean('Parameters','saveNormalized')            
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
            self.convertToNAN = self.conf.getboolean('Parameters','convertToNAN')
            
        except Exception as e:
            print(" Error in reading configuration file:")
            print(f"  {e}\n")

    def createConfig(self):
        try:
            self.dataMLMakerDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
                
        except Exception as e:
            print("Error in creating configuration file:")
            print(f"  {e}\n")
            
    def rescaleList(self, list, value):
        list = [x + value for x in list]
        return list
    
#************************************
# Main
#************************************
def main():
    dP = Conf()
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 DataML_Maker.py <paramFile> <pred column - optional>')
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
        P,V,norm = readParamFile(sys.argv[1], predRCol, rootFile, dP)
    except Exception as e:
        print("\033[1m" + f" Something went wrong during parsing: {e}\n" + "\033[0m")
        return
    
    normTag = ""
    if dP.saveNormalized or dP.normalizeLabel:
        normTag += '_norm'
    
    #************************************
    # Creating training set
    #************************************
    if dP.randomize:
        print(" Creating randomized training set using",dP.minPercVariation*100, "% as max variation on parameters\n")
        Pr = randomize(P, dP)
        
        if dP.fullRandomMatrix:
            randTag = '_fullrand'
        else:
            randTag = '_partrand'
        
        learnRandomFile = learnFile + randTag + str(dP.numRandomAdds) + '_pcVar' + str(int(dP.minPercVariation*100))
        
        learnRandomFile += '_rLab'
        saveLearnFile(Pr, learnRandomFile+normTag, False, dP)
    else:
        saveLearnFile(P, learnFile+normTag, False, dP)

#************************************
# Open Learning Data
#************************************
def readParamFile(paramFile, predRCol, rootFile, dP):

    if dP.fullDataset:
        usecols = range(dP.minCCol,dP.maxCCol)
    else:
        usecols = dP.charCCols
            
    with open(paramFile, 'r') as f:
        df = pd.read_csv(f, delimiter = ",", header=dP.numHeadRows)
    featNames = df.columns.to_list()[dP.numHeadColumns:]
    P2 = df.to_numpy()
    M = np.hstack((P2[:,predRCol],P2[:,usecols])).astype(float)
    
    featNum = np.insert([x -dP.numHeadColumns+1 for x in usecols], 0,0)

    #***************************************
    # Handle Validation File
    #***************************************
    if dP.validFile:
        validFile = rootFile + '_test'
        
        if dP.createRandomValidSet:
            # Global Clean for Random Split
            initial_len = len(M)
            M = np.unique(M, axis=0)
            print(f" Removed {initial_len - len(M)} exact duplicate rows.")
            
            initial_len_2 = len(M)
            non_zero_mask = np.sum(np.abs(M), axis=1) > 1e-6
            M = M[non_zero_mask]
            print(f" Removed {initial_len_2 - len(M)} completely empty/zero rows.\n")
            
            P = np.vstack([featNum, M])
            P, V = formatSubset(P, dP.percentValid, dP.numGroupCols)
            
        else:
            if dP.validRows:
                # 1. Extract manually specified rows based on original index
                V_data = M[dP.validRows, :]
                if V_data.ndim == 1:
                    V_data = V_data.reshape(1, -1) 
                    
                P_data = np.delete(M, dP.validRows, 0)
                
                # 2. Clean the training set 
                initial_len = len(P_data)
                P_data = np.unique(P_data, axis=0)
                print(f" Removed {initial_len - len(P_data)} exact duplicate rows from training set.")
                
                initial_len_2 = len(P_data)
                non_zero_mask = np.sum(np.abs(P_data), axis=1) > 1e-6
                P_data = P_data[non_zero_mask]
                print(f" Removed {initial_len_2 - len(P_data)} completely empty/zero rows from training set.\n")
                
                # 3. Prevent Group Leakage 
                v_base_features = V_data[:, :dP.numGroupCols].astype(float)
                p_base_features = P_data[:, :dP.numGroupCols].astype(float)
                
                leakage_mask = np.zeros(P_data.shape[0], dtype=bool)
                for v_base in v_base_features:
                    match = np.all(np.isclose(p_base_features, v_base, atol=1e-6), axis=1)
                    leakage_mask = leakage_mask | match
                
                P_data_clean = P_data[~leakage_mask]
                
                print(f" Manually extracted {len(dP.validRows)} validation rows.")
                print(f" Removed {np.sum(leakage_mask)} overlapping rows from training to prevent group leakage.")
                
                P = np.vstack([featNum, P_data_clean])
                V = np.vstack([featNum, V_data])
                
    if dP.purgeUndefRows:
        P = purgeRows(P)
    
    if dP.convertToNAN:
        # =========================================================
        # MISSING DATA HANDLING: Convert 0.0 to np.nan in features
        # (Applied to row index 1:end and column index 1:end)
        # =========================================================
        p_features = P[1:, 1:].astype(float)
        p_features[p_features == 0.0] = np.nan
        P[1:, 1:] = p_features
    
        v_features = V[1:, 1:].astype(float)
        v_features[v_features == 0.0] = np.nan
        V[1:, 1:] = v_features
            
    if dP.saveNormalized or dP.normalizeLabel:
        norm = Normalizer(P, dP)
        saveLearnFile(V, validFile+"_norm", False, dP)
        norm.save()
    else:
        norm = None
        saveLearnFile(V, validFile, False, dP)

    return P,V, norm
    
#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile, saveNormFlag, dP):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'w') as f:
                np.savetxt(f, M, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        if saveNormFlag == False:
            print("\n Saving new file (txt) in:", learnFile+"\n")
        else:
            print(" Saving new normalized file (txt) in:", learnFile+"\n")
    else:
        learnFile += '.h5'
        print(" Saving new file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M.astype(np.float64))

#***************************************
# Randomize initial set
#***************************************
def randomize(P, dP):
    Pr = [P]
    if not dP.fullRandomMatrix:
        cols = dP.randomCols
    else:
        cols = P[0,1:].astype(int)
        
    for j in range (0,dP.numRandomAdds):
        rand = randomMatrix(P, cols, dP)
        temp = np.multiply(P[1:],rand)
        Pr.append(temp)
    return np.vstack(Pr)

def randomMatrix(P, cols, dP):
    rand = np.ones(P[1:].shape)
    rand[:,cols] = np.random.uniform(1-dP.minPercVariation,1,rand[:,cols].shape)
    return rand
    
#************************************
# Create validation subset (Group-Aware)
#************************************
def formatSubset(A, percent, num_group_cols):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    
    header = A[0, :]
    data = A[1:, :].astype(float) # Data is already cleaned!
    
    clean_labels = data[:, 0]
    clean_features = data[:, 1:]
    
    base_features = clean_features[:, :num_group_cols]
    unique_samples, groups = np.unique(base_features, axis=0, return_inverse=True)
    
    print(f" Found {len(unique_samples)} unique physical samples.")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=percent, random_state=42)
    train_idx, test_idx = next(gss.split(clean_features, clean_labels, groups=groups))
    
    A_train = clean_features[train_idx]
    Cl_train = clean_labels[train_idx]
    A_cv = clean_features[test_idx]
    Cl_cv = clean_labels[test_idx]
    
    print(" Creating a training set with:", str(len(A_train)), "datapoints")
    print(" Creating a validation set with:", str(len(A_cv)), "datapoints\n")
    
    Atrain = np.vstack((header, np.hstack((Cl_train.reshape(-1, 1), A_train))))
    Atest = np.vstack((header, np.hstack((Cl_cv.reshape(-1, 1), A_cv))))
    
    return Atrain, Atest

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
