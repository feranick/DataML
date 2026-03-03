#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* DataML_Maker
* Adds data from single file to Master Doc
* File must be in ASCII
* version: 2026.03.03.1
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
        self.excludeRows = [x-1 for x in self.excludeRows] # <-- Converted to 0-based index
    
    def dataMLMakerDef(self):
        self.conf['Parameters'] = {
            'saveAsTxt' : True,
            'numHeadColumns' : 1,
            'numHeadRows' : 0,
            'fullDataset' : False,
            'minCCol' : 1,
            'maxCCol' : 28,
            'charCCols' : [21,23,25,27],
            'predRCol' : [29],
            'purgeUndefRows' : False,
            'validFile' : True,
            'createRandomValidSet' : False,
            'numGroupCols' : 5,
            'percentValid' : 0.05,
            'strictNonZeroValid' : False,
            'validRows' : [1,2,3],
            'excludeRows' : [],
            'precData' : 4,
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
            self.strictNonZeroValid = self.conf.getboolean('Parameters','strictNonZeroValid', fallback=False)
            self.validRows = ast.literal_eval(self.dataMLMakerPar['validRows'])
            self.excludeRows = ast.literal_eval(self.dataMLMakerPar['excludeRows'])
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
            
            # <-- REMOVE EXCLUDE ROWS HERE FOR RANDOM SET
            if dP.excludeRows:
                M = np.delete(M, dP.excludeRows, 0)
                print(f" Excluded {len(dP.excludeRows)} specific rows from dataset.")
                
            # Global Clean for Random Split
            initial_len = len(M)
            M = np.unique(M, axis=0)
            print(f" Removed {initial_len - len(M)} exact duplicate rows.")
            
            initial_len_2 = len(M)
            non_zero_mask = np.sum(np.abs(M), axis=1) > 1e-6
            M = M[non_zero_mask]
            print(f" Removed {initial_len_2 - len(M)} completely empty/zero rows.\n")
            
            P = np.vstack([featNum, M])
            P, V = formatSubset(P, dP.percentValid, dP.numGroupCols, dP.strictNonZeroValid)
            
        else:
            if dP.validRows:
                # 1. Extract manually specified rows based on original index
                # <-- ENSURE EXCLUDED ROWS AREN'T IN THE VALIDATION SET
                valid_clean = [r for r in dP.validRows if r not in dP.excludeRows]
                
                V_data = M[valid_clean, :]
                if V_data.ndim == 1:
                    V_data = V_data.reshape(1, -1) 
                    
                # <-- DELETE BOTH VALID AND EXCLUDE ROWS FROM TRAINING SET
                rows_to_drop = list(set(dP.validRows + dP.excludeRows))
                P_data = np.delete(M, rows_to_drop, 0)
                
                if dP.excludeRows:
                    print(f" Excluded {len(dP.excludeRows)} specific rows from dataset.")
                
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
                
                print(f" Manually extracted {len(valid_clean)} validation rows.")
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
def formatSubset(A, percent, num_group_cols, strict_nonzero_valid=False):
    import numpy as np
    
    header = A[0, :]
    data = A[1:, :].astype(float) # Data is already cleaned!
    
    clean_labels = data[:, 0]
    clean_features = data[:, 1:]
    
    base_features = clean_features[:, :num_group_cols]
    unique_samples, groups = np.unique(base_features, axis=0, return_inverse=True)
    
    print(f" Found {len(unique_samples)} unique physical samples (groups).")
    
    if strict_nonzero_valid:
        # 1. Identify rows with ANY zero features
        has_zero_mask = np.any(clean_features == 0, axis=1)
        
        # 2. Identify groups that contain AT LEAST one zero row
        groups_with_zeros = np.unique(groups[has_zero_mask])
        
        total_groups = len(unique_samples)
        ineligible_groups_count = len(groups_with_zeros)
        eligible_groups_count = total_groups - ineligible_groups_count
        
        print(" Strict Non-Zero Validation enabled:")
        print(f"  - {eligible_groups_count} groups eligible for validation (all non-zero).")
        print(f"  - {ineligible_groups_count} groups forced to training (contains zeros).")
        
        # 3. Separate into eligible (Pool B) and ineligible (Pool A) for validation
        eligible_mask = ~np.isin(groups, groups_with_zeros)
        ineligible_mask = np.isin(groups, groups_with_zeros)
        
        pool_b_features = clean_features[eligible_mask]
        pool_b_labels = clean_labels[eligible_mask]
        pool_b_groups = groups[eligible_mask]
        
        pool_a_features = clean_features[ineligible_mask]
        pool_a_labels = clean_labels[ineligible_mask]
        
        # Target validation size based on the ENTIRE dataset ROW count
        target_val_rows = int(len(clean_features) * percent)
        if target_val_rows == 0 and percent > 0:
            target_val_rows = 1
            
        if len(pool_b_features) == 0:
            print(" WARNING: 0 strictly non-zero samples available! Validation set will be empty.")
            A_cv = np.empty((0, clean_features.shape[1]))
            Cl_cv = np.empty((0,))
            A_train = clean_features
            Cl_train = clean_labels
            
        else:
            # We explicitly pick groups until we hit the row target
            unique_b_groups = np.unique(pool_b_groups)
            
            # Shuffle for randomness
            np.random.seed(42)
            np.random.shuffle(unique_b_groups)
            
            selected_groups_for_val = []
            current_val_rows = 0
            
            for g in unique_b_groups:
                g_size = np.sum(pool_b_groups == g)
                selected_groups_for_val.append(g)
                current_val_rows += g_size
                
                # Stop if we've reached or exceeded our row target
                if current_val_rows >= target_val_rows:
                    break
            
            # If target rows exceeded our whole available non-zero pool
            if current_val_rows < target_val_rows:
                print(f" WARNING: Requested validation size ({target_val_rows} rows) exceeds available non-zero data ({current_val_rows} rows). Using all eligible non-zero data.")
            else:
                print(f" Note: Targeted ~{target_val_rows} rows. Picked {len(selected_groups_for_val)} groups yielding {current_val_rows} rows to prevent group splitting.")
            
            val_mask = np.isin(pool_b_groups, selected_groups_for_val)
            train_mask = ~val_mask
            
            A_cv = pool_b_features[val_mask]
            Cl_cv = pool_b_labels[val_mask]
            
            # Training gets all ineligible data (Pool A) + the leftover eligible data
            A_train = np.vstack((pool_a_features, pool_b_features[train_mask]))
            Cl_train = np.concatenate((pool_a_labels, pool_b_labels[train_mask]))
                
    else:
        # Standard Split logic (Iterative selection to match percentage closer)
        target_val_rows = int(len(clean_features) * percent)
        if target_val_rows == 0 and percent > 0:
            target_val_rows = 1
            
        unique_groups = np.unique(groups)
        np.random.seed(42)
        np.random.shuffle(unique_groups)
        
        selected_groups_for_val = []
        current_val_rows = 0
        
        for g in unique_groups:
            g_size = np.sum(groups == g)
            selected_groups_for_val.append(g)
            current_val_rows += g_size
            if current_val_rows >= target_val_rows:
                break
                
        val_mask = np.isin(groups, selected_groups_for_val)
        train_mask = ~val_mask
        
        A_cv = clean_features[val_mask]
        Cl_cv = clean_labels[val_mask]
        A_train = clean_features[train_mask]
        Cl_train = clean_labels[train_mask]
        
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
