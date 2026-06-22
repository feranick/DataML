#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Multi-Perf Regressor
* pyscript version
* version: 2026.06.22.1
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
        self.modelName = "/" + self.modelNameRoot + self.typeDF + self.mode + ".pkl"
        self.summaryFileName = self.modelNameRoot + self.typeDF + self.mode + ".csv"

        self.model_le = self.model_directory + "model_le.pkl"
        self.model_scaling = self.model_directory + "model_scaling.pkl"
        self.model_pca = self.model_directory + "model_encoder.pkl"
        self.norm_file = self.model_directory + "norm_file.pkl"
        self.model_ad = self.model_directory + "model_ad.pkl"
        self.optParFile = "opt_parameters.txt"
        self.rescaleForPCA = False
        self.verbose = 1

    def readConfig(self, configFile):
        try:
            self.conf.read_string(configFile)
            self.typeDF = self.conf.get('Parameters', 'typeDF')
            self.regressor = self.conf.getboolean('Parameters', 'regressor')
            self.normalize = self.conf.getboolean('Parameters', 'normalize')
            self.normalizeLabel = self.conf.getboolean('Parameters', 'normalizeLabel')
        except Exception as e:
            print(" Error in reading configuration file:")
            print(f"  {e}\n")
            # Safe fallbacks
            self.typeDF = self.conf.get('Parameters', 'typeDF', fallback='GradientBoosting')
            self.regressor = self.conf.getboolean('Parameters', 'regressor', fallback=True)
            self.normalize = self.conf.getboolean('Parameters', 'normalize', fallback=False)
            self.normalizeLabel = self.conf.getboolean('Parameters', 'normalizeLabel', fallback=False)


async def getFile(folder, file, bin):
    url = "./" + folder + "/" + file
    if bin:
        data = await fetch(url).bytearray()
    else:
        data = await fetch(url).text()
    return data


def get_selectors():
    """Return list of (perfkey, folder) in DOM (sorted) order."""
    selectors = document.querySelectorAll('#perf-selectors-container select')
    out = []
    for i in range(selectors.length):
        sel = selectors.item(i)
        out.append((sel.getAttribute('data-perfkey'), sel.value))
    return out


async def load_model_bundle(folder):
    """Load model, optional normalizer and AD model for a folder."""
    ini = await getFile(folder, "DataML_DF.ini", False)
    dP = Conf(ini)

    modelPkl = await getFile(folder, dP.modelName, True)
    df = pickle.loads(modelPkl)

    norm = None
    if dP.normalize:
        normPkl = await getFile(folder, dP.norm_file, True)
        norm = pickle.loads(normPkl)

    try:
        model_ad = await getFile(folder, dP.model_ad, True)
        ad_model = pickle.loads(model_ad)
    except:
        ad_model = None

    return dP, df, norm, ad_model


def run_prediction(dP, df, norm, ad_model, R):
    """Predict a single sample R (shape [[...]]). Returns (value, ood_tag)."""
    Rp = np.copy(R)
    if dP.normalize and norm is not None:
        Rp = norm.transform_valid_data(Rp)

    ood = ""
    if ad_model is not None:
        try:
            flags = ad_model.predict(Rp)
            if flags[0] == -1:
                ood = "[OOD]"
        except ValueError as e:
            print(f"   AD check skipped — {e}")

    if dP.normalize and norm is not None:
        pred = norm.transform_inverse_single(df.predict(Rp))
    else:
        pred = df.predict(Rp)

    return float(pred[0]), ood


# ##########################################################
#  Single (manual) prediction across all Perf parameters
# ##########################################################
async def singlePredict(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."
    document.querySelector("#button").disabled = True

    selectors = get_selectors()
    if not selectors:
        output_div.innerText = "No models available."
        document.querySelector("#button").disabled = False
        return

    # Features (shared) read from the first selected model
    first_folder = selectors[0][1]
    features = await getFile(first_folder, "config.txt", False)
    feat_list = [f.strip() for f in features.split(',')]
    
    # --- Verify all selected models share identical features ---
    ok, report = await verify_feature_consistency(selectors, feat_list)
    if not ok:
        output_div.innerText = report
        return

    # Read input values
    R = []
    for i in range(len(feat_list)):
        R.append(document.querySelector("#Entry" + str(i)).value)
    try:
        R = np.array([[float(x) for x in R]])
    except ValueError:
        output_div.innerText = "Error: all input parameters must be valid numbers."
        document.querySelector("#button").disabled = False
        return
    Rorig = np.copy(R)

    results = []
    any_ood = False
    for (perfkey, folder) in selectors:
        try:
            dP, df, norm, ad_model = await load_model_bundle(folder)
            val, ood = run_prediction(dP, df, norm, ad_model, R)
            if ood:
                any_ood = True
            results.append((perfkey, folder, f"{val:.5f}", ood))
        except Exception as e:
            print(f"Error predicting {perfkey} ({folder}): {e}")
            results.append((perfkey, folder, "ERROR", ""))

    # ---- Build HTML output ----
    html = "<b>Input parameters</b><table class='res-table'>"
    html += "<tr><th>Parameter</th><th>Value</th></tr>"
    for i in range(len(feat_list)):
        html += f"<tr><td>{feat_list[i]}</td><td>{Rorig[0][i]}</td></tr>"
    html += "</table><br>"

    html += "<b>Predicted performance parameters</b><table class='res-table'>"
    html += "<tr><th>Perf</th><th>Model</th><th>Predicted value</th><th>AD</th></tr>"
    for (perfkey, folder, val, ood) in results:
        flag = "<span class='ood'>" + ood + "</span>" if ood else ""
        html += f"<tr><td>{perfkey}</td><td>{folder}</td><td>{val}</td><td>{flag}</td></tr>"
    html += "</table>"

    if any_ood:
        html += ("<br><b class='ood'>WARNING: [OOD] parameters fall OUTSIDE the known "
                 "Applicability Domain. Those predictions may be unreliable.</b>")

    output_div.innerHTML = html
    document.querySelector("#button").disabled = False


# ##########################################################
#  Batch prediction (CSV) across all Perf parameters
# ##########################################################
async def batchPredict(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    selectors = get_selectors()
    if not selectors:
        output_div.innerText = "No models available."
        return

    first_folder = selectors[0][1]
    features = await getFile(first_folder, "config.txt", False)
    feat_list = [f.strip() for f in features.split(',')]

    inputFile = document.getElementById("inputFile").files.item(0)
    array_buf = await inputFile.arrayBuffer()
    file_bytes = array_buf.to_bytes()
    csv_file = BytesIO(file_bytes)
    try:
        dataDf = pd.read_csv(csv_file)
    except pd.errors.ParserError:
        output_div.innerText = "File: \"" + inputFile.name + "\" is not a valid CSV or has parsing errors."
        return False
    except pd.errors.EmptyDataError:
        output_div.innerText = "\"" + inputFile.name + "\" is an empty CSV file"
        return False
    except Exception as e:
        print(f"Unexpected error reading '{inputFile.name}': {e}")
        output_div.innerText = "This is not a valid CSV file - see error in log"
        return False
    document.getElementById('inputFile').value = ''

    # Same validation as before: features as rows, samples as columns
    if len(feat_list) != dataDf.shape[0]:
        output_div.innerText = " Please choose the right models for this file (feature count mismatch).\n"
        return 0

    sample_names = [str(dataDf.columns[ci]) for ci in range(1, dataDf.shape[1])]
    perf_keys = []
    # matrix[sample][perfkey] = (value_str, ood)
    matrix = {sn: {} for sn in sample_names}
    any_ood = False

    for (perfkey, folder) in selectors:
        perf_keys.append(perfkey)
        try:
            dP, df, norm, ad_model = await load_model_bundle(folder)
        except Exception as e:
            print(f"Error loading {perfkey} ({folder}): {e}")
            for sn in sample_names:
                matrix[sn][perfkey] = ("ERROR", "")
            continue

        for ci in range(1, dataDf.shape[1]):
            sn = str(dataDf.columns[ci])
            try:
                R = np.array([dataDf.iloc[:, ci].tolist()], dtype=float)
                val, ood = run_prediction(dP, df, norm, ad_model, R)
                if ood:
                    any_ood = True
                matrix[sn][perfkey] = (f"{val:.5f}", ood)
            except ValueError:
                matrix[sn][perfkey] = ("ERROR", "")

    # ---- Build on-screen HTML table (rows = samples, cols = Perf) ----
    html = f"<b>Batch prediction for {inputFile.name}</b><table class='res-table'>"
    html += "<tr><th>Sample</th>" + "".join(f"<th>{pk}</th>" for pk in perf_keys) + "</tr>"
    for sn in sample_names:
        html += f"<tr><td>{sn}</td>"
        for pk in perf_keys:
            val, ood = matrix[sn].get(pk, ("", ""))
            flag = " <span class='ood'>" + ood + "</span>" if ood else ""
            html += f"<td>{val}{flag}</td>"
        html += "</tr>"
    html += "</table>"
    if any_ood:
        html += ("<br><b class='ood'>WARNING: [OOD] entries fall OUTSIDE the known "
                 "Applicability Domain. Those predictions may be unreliable.</b>")
    output_div.innerHTML = html

    # ---- Build downloadable CSV (rows = samples, cols = Perf) ----
    summaryFile = [
        ['File:', inputFile.name],
        ['DataML_DF', 'Multi-Perf'],
        [],
        ['Sample'] + perf_keys,
    ]
    for sn in sample_names:
        row = [sn]
        for pk in perf_keys:
            val, ood = matrix[sn].get(pk, ("", ""))
            row.append(val + (" " + ood if ood else ""))
        summaryFile.append(row)

    await create_csv_download(summaryFile, "Results_" + inputFile.name)


async def create_csv_download(rows, filename="results.csv"):
    download_div = document.getElementById('download-area')
    if download_div is None:
        print("FATAL: Cannot find #download-area.")
        return
    if not rows:
        download_div.innerText = "Error: No data to process for download."
        return
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
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
        download_div.innerText = f"An error occurred during CSV creation: {e}"

async def verify_feature_consistency(selectors, ref_features):
    """
    Verify that every selected model's config.txt matches the reference feature list.
    Returns (ok: bool, report: str).
    'selectors' is a list of (perfkey, folder); 'ref_features' is the reference list.
    """
    mismatches = []
    for (perfkey, folder) in selectors:
        try:
            raw = await getFile(folder, "config.txt", False)
            feats = [f.strip() for f in raw.strip().split(',')]
        except Exception as e:
            mismatches.append(f"  • {perfkey} ({folder}): could not read config.txt — {e}")
            continue

        if feats != ref_features:
            # Build a concise diff description
            if len(feats) != len(ref_features):
                detail = f"count {len(feats)} vs expected {len(ref_features)}"
            else:
                diffs = [f"[{i}] '{feats[i]}' != '{ref_features[i]}'"
                         for i in range(len(feats)) if feats[i] != ref_features[i]]
                detail = "; ".join(diffs)
            mismatches.append(f"  • {perfkey} ({folder}): {detail}")

    if mismatches:
        report = ("ERROR: Input parameter mismatch between models.\n"
                  "All Perf models must share identical config.txt features.\n\n"
                  "Reference (from first model):\n  " + ", ".join(ref_features) +
                  "\n\nMismatched models:\n" + "\n".join(mismatches))
        return False, report
    return True, ""
