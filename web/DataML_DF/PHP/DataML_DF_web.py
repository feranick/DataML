#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***************************************************
* DataML Decision Forests - Classifier and Regressor
* v2025.04.05.2
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
***************************************************
'''
#print(__doc__)

import numpy as np
import sys, os.path, configparser, ast
import pickle
#from libDataML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def DataML_DF_web():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self, folder):
    
        #################################
        ### Types of estimators:
        ### Set using: typeDF
        ### - GradientBoosting (default)
        ### - RandomForest
        ### - HistGradientBoosting
        ### - DecisionTree
        #################################
        
        ###################################
        ### Types of optimization scoring:
        ### Set using:
        ### optScoringR for Regression
        ### optScoringC for Classification
        ### - neg_mean_absolute_error (default)
        ### - r2
        ### - accuracy
        #################################

        self.appName = "DataML_DF"
        confFileName = "DataML_DF.ini"
        self.configFile = os.path.join(os.getcwd(),confFileName)
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
        self.model_directory = "/"
        if self.regressor:
            self.mode = "Regressor"
            self.metric = "MAE"
        else:
            self.mode = "Classifier"
            self.metric = "Accuracy"

        self.modelNameRoot = "model_DF_"
        self.modelName = folder + "/" +self.modelNameRoot + self.typeDF + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDF + self.mode + ".csv"

        self.tb_directory = "model_DF"
        self.model_name = self.model_directory+self.modelNameRoot

        self.model_le = folder + self.model_directory+"model_le.pkl"
        self.model_scaling = folder + self.model_directory+"model_scaling.pkl"
        self.model_pca = folder + self.model_directory+"model_encoder.pkl"
        self.norm_file = folder + self.model_directory+"norm_file.pkl"

        self.optParFile = "opt_parameters.txt"

        self.rescaleForPCA = False
        if self.regressor:
            self.optScoring = self.optScoringR
        else:
            self.optScoring = self.optScoringC

        self.verbose = 1

    def datamlDef(self):
        self.conf['Parameters'] = {
            'typeDF' : 'GradientBoosting',
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
            'normalizeLabel' : False,
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            'plotFeatImportance' : False,
            'optimizeParameters' : False,
            'optScoringR' : 'neg_mean_absolute_error',
            'optScoringC' : 'accuracy',
            }
    
    def sysDef(self):
        self.conf['System'] = {
            'random_state' : 1,
            'n_jobs' : 1
            }
    
    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
    
            self.typeDF = self.conf.get('Parameters','typeDF')
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
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            self.optimizeParameters = self.conf.getboolean('Parameters','optimizeParameters')
            self.optScoringR = self.conf.get('Parameters','optScoringR')
            self.optScoringC = self.conf.get('Parameters','optScoringC')
            self.random_state = ast.literal_eval(self.sysDef['random_state'])
            self.n_jobs = self.conf.getint('System','n_jobs')
            
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.datamlDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    #try:
    predict(sys.argv[1], sys.argv[2], sys.argv[3])
    #except:
    #     print("Select model and run prediction.")

#************************************
# Prediction - backend
#************************************
def getPrediction(dP, df, R, le):
    if dP.normalize:
        try:
            with open(dP.norm_file, "rb") as f:
                norm = pickle.load(f)
                R = norm.transform_valid_data(R)
        except:
            print("\033[1m pkl file not found \033[0m")
            return

    if dP.runDimRedFlag:
        R = runPCAValid(R, dP)

    if dP.regressor:
        if dP.normalize:
            pred = norm.transform_inverse_single(df.predict(R))
        else:
            pred = df.predict(R)
        pred_classes = None
        proba = None
    else:
        pred = le.inverse_transform_bulk(df.predict(R))

        if dP.normalize:
            pred_classes = norm.transform_inverse(np.asarray(le.inverse_transform_bulk(df.classes_)))
        else:
            pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba(R)

    return pred, pred_classes, proba

#************************************
# Prediction - frontend
#************************************
def predict(folder, testFile, features):
    dP = Conf(folder)

    #R, _ = readTestFile(testFile)

    R = np.array([testFile.split(',')], dtype=float)
    feat = features.split(',')

    if any(item == '' for item in R[0]):
        print("  Make sure all entries are filled before running the prediction")
        return 0

    if dP.runDimRedFlag:
        R = runPCAValid(R, dP)

    with open(dP.modelName, "rb") as f:
        df = pickle.load(f)

    if dP.regressor:
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)

    pred, pred_classes, proba = getPrediction(dP, df, R, le)

    print('\n==================================')
    if dP.regressor:
        print(" {0:s} = {1:.2f}  ".format(folder[:5], pred[0]))
    else:
        print(' Prediction\t| Probability [%]')
        print('----------------------------------')
        ind = np.where(proba[0]==np.max(proba[0]))[0]
        for j in range(len(ind)):
            print(" {0:.2f}\t\t| {1:.2f} ".format(pred_classes[ind[j]], 100*proba[0][ind[j]]))
    print('==================================')
    for i in range(0,len(R[0])):
        print(" {0:s} = {1:s}  ".format(feat[i], str(R[0][i])))
    print('==================================')
    print('',dP.typeDF,dP.mode)
    print('==================================\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
