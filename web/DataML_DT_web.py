#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* DataML Decision Trees - Classifier and Regressor
* v2024.10.16.4
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
***************************************************
'''
#print(__doc__)

import numpy as np
import sys, os.path, configparser
import platform, pickle, h5py, csv, glob, math
#from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_DT():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
    
        #############################
        ### Types of estimators:
        ### - RandomForest
        ### - HistGradientBoosting
        ### - GradientBoosting
        ### - DecisionTree
        #############################
        self.appName = "DataML_DT"
        confFileName = "DataML_DT.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist. Please create one with DataML_DT.py\n")
        self.readConfig(self.configFile)
        self.model_directory = "./"
        if self.regressor:
            self.mode = "Regressor"
            self.metric = "MAE"
        else:
            self.mode = "Classifier"
            self.metric = "Accuracy"
        
        self.modelNameRoot = "model_DT_"
        self.modelName = self.modelNameRoot + self.typeDT + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDT + self.mode + ".csv"
        
        self.tb_directory = "model_DT"
        self.model_name = self.model_directory+self.modelNameRoot
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.model_scaling = self.model_directory+"model_scaling.pkl"
        self.model_pca = self.model_directory+"model_encoder.pkl"
        self.norm_file = self.model_directory+"norm_file.pkl"
                    
        self.rescaleForPCA = False
            
    def datamlDef(self):
        self.conf['Parameters'] = {
            'typeDT' : 'GradientBoosting',
            'regressor' : False,
            'n_estimators' : 4,
            'max_depth' : 7,
            'max_features' : 0.5,
            'epochs' : 100,
            'l_rate' : 0.1,
            'cv_split' : 0.05,
            'trainFullData' : True,
            'fullSizeBatch' : False,
            'batch_size' : 8,
            'numLabels' : 1,
            'normalize' : False,
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            'plotFeatImportance' : False,
            }
    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 3,
            'n_jobs' : 1
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
        
            self.typeDT = self.conf.get('Parameters','typeDT')
            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.n_estimators = self.conf.getint('Parameters','n_estimators')
            self.max_depth = self.conf.getint('Parameters','max_depth')
            self.max_features = self.conf.getfloat('Parameters','max_features')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.trainFullData = self.conf.getboolean('Parameters','trainFullData')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.n_jobs = self.conf.getint('System','n_jobs')
            
        except:
            print(" Error in reading configuration file. Please check it\n")

#************************************
# Main
#************************************
def main():
    dP = Conf()
    predict(sys.argv[1], None)

#************************************
# Prediction
#************************************
def predict(testFile, normFile):
    dP = Conf()
    
    R, _ = readTestFile(testFile)

    if normFile is not None:
        try:
            with open(normFile, "rb") as f:
                norm = pickle.load(f)
            print("  Opening pkl file with normalization data:",normFile)
            print("  Normalizing validation file for prediction...\n")
            R = norm.transform_valid_data(R)
        except:
            print("\033[1m pkl file not found \033[0m")
            return
     
    if dP.runDimRedFlag:
        R = runPCAValid(R, dP)
            
    with open(dP.modelName, "rb") as f:
        dt = pickle.load(f)
    
    if dP.regressor:
        pred = dt.predict(R)
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        pred = le.inverse_transform_bulk(dt.predict(R))
        pred_classes = le.inverse_transform_bulk(dt.classes_)
        proba = dt.predict_proba(R)
        
    print('\n  ================================================================================')
    print('  \033[1m',dP.typeDT,dP.mode,'\033[0m')
    print('  ================================================================================')
    if dP.regressor:
        print('   Filename\t\t| Prediction')
        print('  --------------------------------------------------------------------------------')
        print("   {0:s}\t| {1:.2f}  ".format(testFile, pred[0]))
    else:
        print('   Filename\t\t| Prediction\t| Probability')
        print('  --------------------------------------------------------------------------------')
        ind = np.where(proba[0]==np.max(proba[0]))[0]
        for j in range(len(ind)):
            print("   {0:s}\t| {1:.2f}\t| {2:.2f} ".format(testFile, pred_classes[ind[j]], 100*proba[0][ind[j]]))
        print("")
    print('  ================================================================================\n')

#************************************
# Open Testing Data
#************************************
def readTestFile(testFile):
    try:
        with open(testFile, 'r') as f:
            #print("\n  Opening sample data for prediction:",testFile)
            Rtot = np.loadtxt(f, unpack =True)
        R=np.array([Rtot[1,:]])
        Rx=np.array([Rtot[0,:]])
    except:
        print("\033[1m\n File not found or corrupt\033[0m\n")
        return 0, False
    return R, True

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
