#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* pyscript version
* v2024.11.07.1
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import numpy as np
import sys, configparser
from pyscript import fetch, document
import js
import _pickle as pickle
#import pickle

baseUrl = "https://www.example.com"

class Conf():
    def __init__(self, folder, ini):
    
        #############################
        ### Types of estimators:
        ### - RandomForest
        ### - HistGradientBoosting
        ### - GradientBoosting
        ### - DecisionTree
        #############################

        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        
        self.readConfig(ini)
        self.model_directory = "/"
        if self.regressor:
            self.mode = "Regressor"
            self.metric = "MAE"
        else:
            self.mode = "Classifier"
            self.metric = "Accuracy"
        
        self.modelNameRoot = "model_DF_"
        self.modelName = "/" +self.modelNameRoot + self.typeDF + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDF + self.mode + ".csv"
        
        self.tb_directory = "model_DF"
        self.model_name = self.model_directory+self.modelNameRoot
        
        self.model_le = folder + self.model_directory+"model_le.pkl"
        self.model_scaling = folder + self.model_directory+"model_scaling.pkl"
        self.model_pca = folder + self.model_directory+"model_encoder.pkl"
        self.norm_file = folder + self.model_directory+"norm_file.pkl"
                    
        self.rescaleForPCA = False
            
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
            'runDimRedFlag' : False,
            'typeDimRed' : 'SparsePCA',
            'numDimRedComp' : 3,
            'plotFeatImportance' : False,
            }
    def sysDef(self):
        self.conf['System'] = {
            'random_state' : None,
            'n_jobs' : 1
            }

    def readConfig(self,configFile):
        try:
            self.conf.read_string(configFile)
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
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            
            self.random_state = eval(self.sysDef['random_state'])
            self.n_jobs = self.conf.getint('System','n_jobs')
            
        except:
            print(" Error in reading configuration file. Please check it\n")

async def getFile(folder, file, bin):
    url = baseUrl+folder+"/"+file
    if bin:
        data = await fetch(url).bytearray()
    else:
        data = await fetch(url).text()
    return data
    
async def getModel(event):
    global df
    document.querySelector("#button").disabled = True
    document.querySelector("#model").disabled = True
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Loading ML model..."
    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    dP = Conf(folder, ini)
    modelPkl = await getFile(folder, dP.modelName, True)
    df = pickle.loads(modelPkl)
    output_div.innerHTML = ""
    document.querySelector("#button").disabled = False
    document.querySelector("#model").disabled = False
    
async def main(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    features = await getFile(folder, "config.txt", False)
    dP = Conf(folder, ini)
    
    # Use this when opening modelPkl here.
    #modelPkl = await getFile(folder, dP.modelName, True)
    #df = pickle.loads(modelPkl)
    
    # Use this when opening modelPkl in JS
    #df = pickle.loads(js.modelPkl.to_py())
    
    input_text = document.querySelector("#Entry0")
    #output_div.innerText = "The Value for Entry0:", input_text.value;
    R = []
    for i in range(len(features.split(','))):
        R.append(int(document.querySelector("#Entry"+str(i)).value))
    
    output = '============================\n '
    if dP.regressor:
        pred = df.predict([R])
        proba = ""
        output += folder[:5] +" = " + str(pred[0])[:5]
    else:
        lePkl = await getFile("", dP.model_le, True)
        le = pickle.loads(lePkl)
        pred = le.inverse_transform_bulk(df.predict([R]))
        pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba([R])
        ind = np.where(proba[0]==np.max(proba[0]))[0]
        
        output += ' Prediction\t| Probability [%]\n'
        output += '------------------------------------'
        for j in range(len(ind)):
            output += "\n" + str(pred_classes[ind[j]]) + "\t\t|  " + str(100*proba[0][ind[j]])[:5]
            
    output += '\n============================'
    for i in range(0,len(R)):
        output += "\n "+ features.split(',')[i].strip() + " = " + str(R[i])
    output += '\n============================'
    output += '\n'+dP.typeDF+" "+dP.mode
    output += '\n============================\n'
    output_div.innerText = output

