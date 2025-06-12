#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* pyscript version
* v2025.06.12.1
* Uses: sklearn
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import numpy as np
import pandas as pd
from io import BytesIO
import sys, configparser, ast, io, csv
import asyncio
from pyscript import document
from js import window, Blob, URL, fetch

# Use standard library pickle, which is aliased by pyscript
import pickle
from libDataML import *


# Global variable to hold the loaded model
df = None


class Conf():
    def __init__(self, configIni):
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
        except Exception as e:
            print(f"Error in reading configuration file: {e}\n")

async def getFile(folder, file, binary):
    """
    Fetches a file from a specified folder.
    """
    url = f"./{folder}/{file}"
    response = await fetch(url)
    if not response.ok:
        raise Exception(f"Failed to fetch {url}, status: {response.status}")

    if binary:
        js_array_buffer = await response.arrayBuffer();
        return js_array_buffer.to_bytes()
    else:
        return await response.text()

async def load_py_model():
    """
    Loads the Python parts of the model (ini, pkl) into memory.
    This is called AFTER the JS has set up the UI.
    """
    global df
    output_div = document.querySelector("#output")
    button = document.querySelector("#button")
    input_file = document.getElementById("inputFile")
    model_dropdown = document.querySelector("#model")

    # Disable UI elements during Python model loading
    button.disabled = True
    input_file.disabled = True
    output_div.innerHTML = "Loading ML model..."
    
    folder = model_dropdown.value
    print(f"Python: Loading model from folder '{folder}'")
    
    try:
        ini_content = await getFile(folder, "DataML_DF.ini", False)
        dP = Conf(ini_content)
        model_pkl_bytes = await getFile(folder, dP.modelName, True)
        
        # Load the model using pickle
        df = pickle.loads(model_pkl_bytes)
        print(f"Python: Model '{dP.modelName}' loaded successfully.")
        
    except Exception as e:
        error_msg = f"Python Error: Failed to load model '{folder}'. {type(e).__name__}: {e}"
        print(error_msg)
        if output_div: output_div.innerHTML = f'<span style="color: red;">{error_msg}</span>'
        # Keep UI disabled if the model fails to load, as predict will not work
        return
    
    # Re-enable UI elements after successful load
    output_div.innerHTML = ""
    button.disabled = False
    input_file.disabled = False


async def on_model_select(event):
    """
    This function is the main event handler for the model dropdown.
    It orchestrates the synchronized loading process.
    """
    print("Python: User selected a new model. Starting update process...")
    
    # 1. Call the JavaScript function to update the UI (features, links, etc.)
    #    and wait for it to complete. It returns true on success.
    js_success = await window.js_selectModel()
    
    # 2. If the JavaScript part succeeded, load the corresponding Python model.
    if js_success:
        print("Python: JavaScript UI setup complete. Now loading Python model.")
        await load_py_model()
    else:
        print("Python: JavaScript UI setup failed. Aborting Python model load.")


async def singlePredict(event):
    """
    Runs a single prediction based on user input fields.
    Assumes the model is already loaded in the global 'df'.
    """
    if not df:
        document.querySelector("#output").innerHTML = '<span style="color: red;">Model not loaded. Please select a model first.</span>'
        return
        
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait, predicting..."

    folder = document.querySelector("#model").value
    # Fetch config files needed for this prediction
    ini = await getFile(folder, "DataML_DF.ini", False)
    features_str = await getFile(folder, "config.txt", False)
    features = features_str.split(',')
    dP = Conf(ini)

    # Gather input values from the dynamically created entry fields
    R = [document.querySelector(f"#Entry{i}").value for i in range(len(features))]
    R = np.array([[float(char) for char in R]])
    Rorig = np.copy(R)

    output = '============================\n'

    if dP.normalize:
        norm_pkl = await getFile(folder, dP.norm_file, True)
        norm = pickle.loads(norm_pkl)
        R = norm.transform_valid_data(R)

    if dP.regressor:
        pred = norm.transform_inverse_single(df.predict(R)) if dP.normalize else df.predict(R)
        output += f"{folder[:5]} = {str(pred[0])[:5]}"
    else:
        le_pkl = await getFile(folder, dP.model_le, True)
        le = pickle.loads(le_pkl)
        pred = le.inverse_transform_bulk(df.predict(R))
        pred_classes = le.inverse_transform_bulk(df.classes_)
        proba = df.predict_proba(R)
        ind = np.where(proba[0] == np.max(proba[0]))[0]
        output += ' Prediction\t| Probability [%]\n'
        output += '------------------------------------'
        for j in ind:
            p_class = str(round(norm.transform_inverse_single(pred_classes[j]), 2)) if dP.normalize else str(round(pred_classes[j], 2))
            output += f"\n{p_class}\t\t|  {str(100 * proba[0][j])[:5]}"

    output += '\n============================'
    for i in range(len(Rorig[0])):
        output += f"\n {features[i].strip()} = {Rorig[0][i]}"
    output += f'\n============================\n{dP.typeDF} {dP.mode}\n============================\n'
    output_div.innerText = output


async def batchPredict(event):
    """
    Runs batch prediction from an uploaded file.
    Assumes the model is already loaded in the global 'df'.
    """
    if not df:
        document.querySelector("#output").innerHTML = '<span style="color: red;">Model not loaded. Please select a model first.</span>'
        return

    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait, processing file..."
    
    # ... The rest of your batchPredict logic remains largely the same ...
    # It correctly re-fetches its required config/normalization files.
    # Just ensure the global `df` is used instead of reloading it.

    folder = document.querySelector("#model").value
    ini = await getFile(folder, "DataML_DF.ini", False)
    features = await getFile(folder, "config.txt", False)
    dP = Conf(ini)

    if dP.normalize:
        normPkl = await getFile(folder, dP.norm_file, True)
        norm = pickle.loads(normPkl)

    inputFile = document.getElementById("inputFile").files.item(0)
    array_buf = await inputFile.arrayBuffer()
    file_bytes = await array_buf.to_bytes() # Use await for pyodide >= 0.23
    csv_file = BytesIO(file_bytes)
    dataDf = pd.read_csv(csv_file)
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
        # Correctly fetching le file for batch processing
        lePkl = await getFile(folder, dP.model_le, True)
        le = pickle.loads(lePkl)
        summaryFile = np.vstack((summaryFile,['Sample','Predicted Value','Probability %']))

    for i in range(1,dataDf.shape[1]):
        R = np.array([dataDf.iloc[:,i].tolist()], dtype=float)

        if dP.normalize:
            R = norm.transform_valid_data(R)

        if dP.regressor:
            pred = norm.transform_inverse_single(df.predict(R)) if dP.normalize else df.predict(R)
            output += f"\n {dataDf.columns[i]} = {str(pred[0])[:5]}"
            summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred[0],'']))
        else:
            pred = le.inverse_transform_bulk(df.predict(R))
            pred_classes = le.inverse_transform_bulk(df.classes_)
            proba = df.predict_proba(R)
            ind = np.where(proba[0]==np.max(proba[0]))[0]

            for j in range(len(ind)):
                p_class = str(round(norm.transform_inverse_single(pred_classes[ind[j]]),2)) if dP.normalize else str(round(pred_classes[ind[j]],2))
                output += f"\n {dataDf.columns[i]} = {p_class}\t\t|  {str(100*proba[0][ind[j]])[:5]}%)"
                summaryFile = np.vstack((summaryFile,[dataDf.columns[i],pred_classes[ind[j]],round(100*proba[0][ind[j]],1)]))

    output += '\n======================================='
    output += f'\n{dP.typeDF} {dP.mode}'
    output += '\n=======================================\n'
    output_div.innerText = output
    await create_csv_download(summaryFile,None,f"Results_{inputFile.name}")

# The create_csv_download function remains unchanged
async def create_csv_download(numpy_array, headers=None, filename="results.csv"):
    download_div = document.getElementById('download-area')
    if download_div is None:
        error_message = "FATAL ERROR: Cannot find HTML element with id='download-area'."
        print(error_message)
        output_div = document.querySelector("#output")
        if output_div:
            output_div.innerText += f"\n\n{error_message}"
        return
    if numpy_array is None or numpy_array.size == 0:
        download_div.innerText = "Error: No data to process for download."
        return
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        if headers:
            if isinstance(headers, (list, tuple)):
                writer.writerow(headers)
        writer.writerows(numpy_array)
        csv_string = output.getvalue()
        output.close()
        blob = Blob.new([csv_string], {type: 'text/csv;charset=utf-8;'})
        url = URL.createObjectURL(blob)
        link = document.createElement('a')
        link.href = url
        link.download = filename
        link.textContent = f'Download {filename}'
        download_div.innerHTML = ''
        download_div.appendChild(link)
    except Exception as e:
        error_message = f"An error occurred during CSV creation/linking: {e}"
        print(error_message)
        download_div.innerText = error_message

async def main():
    """
    Main entry point for application setup. Runs once when PyScript is ready.
    """
    print("Python: Runtime is ready. Initializing application...")
    
    # 1. Call JS to initialize the UI, like setting the dropdown
    #    to its last saved state from cookies.
    window.js_init()
    
    # 2. Call the main selection function to load the UI and the model
    #    for the default/initial selection.
    await on_model_select(None)
    
    print("Python: Initial setup complete. Application is ready.")

# This is the new entry point that starts the entire application.
asyncio.ensure_future(main())
