#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* pyscript version
* v2026.01.22.1
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import numpy as np
import pandas as pd
from io import BytesIO
import sys, configparser, ast, io, csv
from js import document, Blob, URL
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
            'featureReduction' : False,
            'minNumFeatures' : 4,
            }
    def sysDef(self):
        self.conf['System'] = {
            'random_state' : None,
            'n_jobs' : 1,
            'saveAsTxt' : True,
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
            self.normalizeLabel = self.conf.getboolean('Parameters','normalizeLabel')
            self.runDimRedFlag = self.conf.getboolean('Parameters','runDimRedFlag')
            self.typeDimRed = self.conf.get('Parameters','typeDimRed')
            self.numDimRedComp = self.conf.getint('Parameters','numDimRedComp')
            self.plotFeatImportance = self.conf.getboolean('Parameters','plotFeatImportance')
            self.optimizeParameters = self.conf.getboolean('Parameters','optimizeParameters')
            self.optScoringR = self.conf.get('Parameters','optScoringR')
            self.optScoringC = self.conf.get('Parameters','optScoringC')
            self.featureReduction = self.conf.getboolean('Parameters','featureReduction')
            self.minNumFeatures = self.conf.getint('Parameters','minNumFeatures')
            self.random_state = ast.literal_eval(self.sysDef['random_state'])
            self.n_jobs = self.conf.getint('System','n_jobs')
            self.saveAsTxt = self.conf.getboolean('System','saveAsTxt')

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
    try:
        ini = await getFile(folder, "DataML_DF.ini", False)
        dP = Conf(ini)
        modelPkl = await getFile(folder, dP.modelName, True)
        df = pickle.loads(modelPkl)
    except pickle.UnpicklingError as e:
        error_msg = f"Error unpickling model '{folder}': {e}"
        print(error_msg)
        if output_div: output_div.innerHTML = error_msg
        return # Stop further execution for this model load
    except MemoryError as e:
        error_msg = f"MemoryError loading model '{folder}': {e}"
        print(error_msg)
        if output_div: output_div.innerHTML = error_msg
        return
    except Exception as e:
        error_msg = f"Generic error loading model '{folder}': {type(e).__name__} - {e}"
        print(error_msg)
        if output_div: output_div.innerHTML = error_msg
        return
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
    modelPkl = await getFile(folder, dP.modelName, True)
    df = pickle.loads(modelPkl)

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
    #dataDf = pd.read_csv(csv_file)
    try:
        dataDf = pd.read_csv(csv_file)
    except pd.errors.ParserError:
        output = "File: \""+inputFile.name+"\" is not a valid CSV or has parsing errors."
        output_div.innerText = output
        return False
    except pd.errors.EmptyDataError:
        output = "\""+inputFile.name+"\" is an empty CSV file"
        output_div.innerText = output
        return False
    except Exception as e:
        output = "This is not a valid CSV file - see error in log"
        print(f"An unexpected error occurred while checking '{inputFile.name}': {e}")
        output_div.innerText = output
        return False
    document.getElementById('inputFile').value = ''

    if len(features.split(',')) != dataDf.shape[0]:
        output = ' Please choose the right model for this file. \n'
        output_div.innerText = output
        return 0

    output = '======================================\n'
    output += ' Prediction for ' + folder[:5]
    output += '\n======================================'

    summaryFile = np.array([['File:',inputFile.name,''],
	['DataML_DF',dP.typeDF,dP.mode],
	['Model',folder,'']])
    if dP.regressor:
        summaryFile = np.vstack((summaryFile,['Sample','Predicted Value','']))
        le = None
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.vstack((summaryFile,['Sample','Predicted Value','Probability %']))

    for i in range(1,dataDf.shape[1]):
        R = np.array([dataDf.iloc[:,i].tolist()], dtype=float)
        Rorig = np.copy(R)

        if dP.normalize:
            R = norm.transform_valid_data(R)

        if dP.regressor:
            try:
                if dP.normalize:
                    pred = norm.transform_inverse_single(df.predict(R))
                else:
                    pred = df.predict(R)
            except ValueError as e:
                output = "Check \""+inputFile.name+"\" for errors or missing values"
                output_div.innerText = output
                return False
            proba = ""
            output += "\n " + dataDf.columns[i] + " = "  + str(pred[0])[:5]
            summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred[0],'']))
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
                summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred_classes[ind[j]],round(100*proba[0][ind[j]],1)]))

    output += '\n======================================='
    output += '\n'+dP.typeDF+" "+dP.mode
    output += '\n=======================================\n'
    output_div.innerText = output
    await create_csv_download(summaryFile,None,"Results_"+inputFile.name)


async def create_csv_download(numpy_array, headers=None, filename="results.csv"):
    # --- Try to get the target element FIRST ---
    download_div = document.getElementById('download-area') # Use js.document if js is imported

    # --- CRUCIAL CHECK ---
    if download_div is None:
        error_message = "FATAL ERROR: Cannot find HTML element with id='download-area'. Download link cannot be created. Please check the HTML file."
        print(error_message)
        # Try to display the error in the main output div if possible
        output_div = document.querySelector("#output") # Assumes an element with id="output" exists
        if output_div:
            # Append error to existing content or set it
            output_div.innerText += f"\n\n{error_message}"
        return # Stop execution, cannot proceed

    # --- Now proceed with the rest of the logic, knowing download_div exists ---
    if numpy_array is None or numpy_array.size == 0:
        print("Input NumPy array is empty or None.")
        # Display the error message INSIDE the verified download_div
        download_div.innerText = "Error: No data to process for download."
        return

    try:
        # Use io.StringIO to act like an in-memory file
        output = io.StringIO()
        writer = csv.writer(output)

        if headers:
            if isinstance(headers, (list, tuple)):
                 writer.writerow(headers)
            else:
                 print("Warning: Headers should be a list or tuple.")

        writer.writerows(numpy_array)
        csv_string = output.getvalue()
        output.close()

        # Create Blob, URL, and Link
        blob = Blob.new([csv_string], {type: 'text/csv;charset=utf-8;'}) # Use js.Blob if js is imported
        url = URL.createObjectURL(blob) # Use js.URL if js is imported
        link = document.createElement('a') # Use js.document if js is imported
        link.href = url
        link.download = filename
        link.textContent = f'Download {filename}'

        # Add the link to the webpage (we know download_div exists)
        download_div.innerHTML = '' # Clear previous content before adding link
        download_div.appendChild(link)

    except Exception as e:
         error_message = f"An error occurred during CSV creation/linking: {e}"
         print(error_message)
         # Display error within the verified download_div
         download_div.innerText = error_message
