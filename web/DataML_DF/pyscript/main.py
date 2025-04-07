#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* pyscript version
* v2025.04.07.1
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import numpy as np
import pandas as pd
from io import BytesIO
import sys, configparser, ast
from pyscript import fetch, document
import _pickle as pickle
from libDataML import *
#import pickle

class Conf():
    def __init__(self, configIni):

        #############################
        ### Types of estimators:
        ### - RandomForest
        ### - HistGradientBoosting
        ### - GradientBoosting
        ### - DecisionTree
        #############################

        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str

        self.readConfig(configIni)
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

        self.model_le = self.model_directory+"model_le.pkl"
        self.model_scaling = self.model_directory+"model_scaling.pkl"
        self.model_pca = self.model_directory+"model_encoder.pkl"
        self.norm_file = self.model_directory+"norm_file.pkl"

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
            'random_state' : None,
            'n_jobs' : 1
            }

    def readConfig(self,configFile):
        try:
            self.conf.read_string(configFile)
            self.datamlPar = self.conf['Parameters']
            self.sysPar = self.conf['System']

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
            self.random_state = ast.literal_eval(self.sysPar['random_state'])
            self.n_jobs = self.conf.getint('System','n_jobs')
        except:
            print(" Error in reading configuration file. Please check it\n")


async def getFile(folder, file, bin):
    url = "./"+folder+"/"+file
    if bin:
        data = await fetch(url).bytearray()
    else:
        data = await fetch(url).text()
    return data

async def getModel(event):
    global df
    document.querySelector("#button").disabled = True
    document.querySelector("#model").disabled = True
    document.getElementById("inputFile").disabled = True
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Loading ML model..."
    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    dP = Conf(ini)
    modelPkl = await getFile(folder, dP.modelName, True)
    df = pickle.loads(modelPkl)
    output_div.innerHTML = ""
    document.querySelector("#button").disabled = False
    document.querySelector("#model").disabled = False
    document.getElementById("inputFile").disabled = False

async def singlePredict(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    features = await getFile(folder, "config.txt", False)
    dP = Conf(ini)

    # Use this when opening modelPkl here.
    #modelPkl = await getFile(folder, dP.modelName, True)
    #df = pickle.loads(modelPkl)

    # Use this when opening modelPkl in JS
    #import js
    #df = pickle.loads(js.modelPkl.to_py())

    R = []
    for i in range(len(features.split(','))):
        R.append(document.querySelector("#Entry"+str(i)).value)
    R = np.array([[float(char) for char in R]])
    Rorig = np.copy(R)
    print(R)
    output = '============================\n '

    if dP.normalize:
        normPkl = await getFile(folder, dP.norm_file, True)
        norm = pickle.loads(normPkl)
        R = norm.transform_valid_data(R)

    if dP.regressor:
        if dP.normalize:
            pred = norm.transform_inverse_single(df.predict(R))
        else:
            pred = df.predict(R)
        proba = ""
        output += folder[:5] +" = " + str(pred[0])[:5]
    else:
        lePkl = await getFile(folder, dP.model_le, True)
        le = pickle.loads(lePkl)
        pred = le.inverse_transform_bulk(df.predict(R))
        pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba(R)
        ind = np.where(proba[0]==np.max(proba[0]))[0]

        output += ' Prediction\t| Probability [%]\n'
        output += '------------------------------------'
        for j in range(len(ind)):
            if dP.normalize:
                p_class = str(round(norm.transform_inverse_single(pred_classes[ind[j]]),2))
            else:
                p_class = str(round(pred_classes[ind[j]],2))
            output += "\n" + p_class + "\t\t|  " + str(100*proba[0][ind[j]])[:5]

    output += '\n============================'
    for i in range(0,len(Rorig[0])):
        output += "\n "+ features.split(',')[i].strip() + " = " + str(Rorig[0][i])
    output += '\n============================'
    output += '\n'+dP.typeDF+" "+dP.mode
    output += '\n============================\n'
    output_div.innerText = output


async def batchPredict(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    features = await getFile(folder, "config.txt", False)
    dP = Conf(ini)

    # Use this when opening modelPkl here.
    #modelPkl = await getFile(folder, dP.modelName, True)
    #df = pickle.loads(modelPkl)

    # Use this when opening modelPkl in JS
    #import js
    #df = pickle.loads(js.modelPkl.to_py())

    if dP.normalize:
        normPkl = await getFile(folder, dP.norm_file, True)
        norm = pickle.loads(normPkl)

    inputFile = document.getElementById("inputFile").files.item(0)
    array_buf = await inputFile.arrayBuffer()
    file_bytes = array_buf.to_bytes()
    csv_file = BytesIO(file_bytes)
    dataDf = pd.read_csv(csv_file)
    document.getElementById('inputFile').value = ''

    if len(features.split(',')) != dataDf.shape[1]-1:
        output = ' Please choose the right model for this file. \n'
        output_div.innerText = output 
        return 0

    output = '======================================\n'
    output += ' Prediction for ' + folder[:5]
    output += '\n======================================'

    for i in range(1,dataDf.shape[1]):
        R = np.array([dataDf.iloc[:,i].tolist()])
        Rorig = np.copy(R)

        if dP.normalize:
            R = norm.transform_valid_data(R)

        if dP.regressor:
            if dP.normalize:
                pred = norm.transform_inverse_single(df.predict(R))
            else:
                pred = df.predict(R)
            proba = ""
            output += "\n " + dataDf.columns[i] + " = "  + str(pred[0])[:5]
        else:
            lePkl = await getFile(folder, dP.model_le, True)
            le = pickle.loads(lePkl)
            pred = le.inverse_transform_bulk(df.predict(R))
            pred_classes = le.inverse_transform_bulk(df.classes_)
            proba = df.predict_proba(R)
            ind = np.where(proba[0]==np.max(proba[0]))[0]

            for j in range(len(ind)):
                if dP.normalize:
                    p_class = str(round(norm.transform_inverse_single(pred_classes[ind[j]]),2))
                else:
                    p_class = str(round(pred_classes[ind[j]],2))
                output += "\n " + dataDf.columns[i] + " = " + p_class + "\t\t|  " + str(100*proba[0][ind[j]])[:5] + "%)"

    output += '\n======================================='
    output += '\n'+dP.typeDF+" "+dP.mode
    output += '\n=======================================\n'
    output_div.innerText = output
