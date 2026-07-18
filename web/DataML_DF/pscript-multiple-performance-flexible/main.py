#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Multi-Perf Regressor
* pyscript version - SUPERSET (name-matched features)
* version: 2026.07.18.1
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


def get_feature_map():
    """
    Read the shared (superset) input fields.
    Each input's `name` attribute is its feature name (set by ml.js).
    Returns (fmap, order):
      fmap  = {feature_name: raw_string_value}
      order = feature names in DOM/creation order (for display)
    """
    inputs = document.querySelectorAll('#feature-entries-container input')
    fmap = {}
    order = []
    for i in range(inputs.length):
        el = inputs.item(i)
        name = el.name
        fmap[name] = el.value
        order.append(name)
    return fmap, order


async def get_model_features(folder):
    """Return this model's feature-name list (in the order the model expects)."""
    raw = await getFile(folder, "config.txt", False)
    return [f.strip() for f in raw.strip().split(',') if f.strip()]


def build_R_from_map(model_feats, fmap):
    """
    Assemble a single-sample array R (shape [[...]]) for `model_feats`
    by pulling values from the superset fmap, in the model's own order.
    Returns (R_or_None, error_message_or_None).
    """
    missing = [f for f in model_feats if f not in fmap]
    if missing:
        return None, "missing features: " + ", ".join(missing)
    vals = []
    for f in model_feats:
        try:
            vals.append(float(fmap[f]))
        except (ValueError, TypeError):
            return None, f"non-numeric value for '{f}': '{fmap[f]}'"
    return np.array([vals]), None


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
    try:
        selectors = get_selectors()
        if not selectors:
            output_div.innerText = "No models available."
            return

        # Shared superset of inputs (name -> value)
        fmap, union_order = get_feature_map()
        if not fmap:
            output_div.innerText = "No input fields available."
            return

        results = []
        any_ood = False

        for (perfkey, folder) in selectors:
            try:
                model_feats = await get_model_features(folder)
            except Exception as e:
                print(f"Error reading config for {perfkey} ({folder}): {e}")
                results.append((perfkey, folder, "ERROR (config)", ""))
                continue

            # Pull just this model's features, by name, from the superset
            R, err = build_R_from_map(model_feats, fmap)
            if err:
                results.append((perfkey, folder, "ERROR: " + err, ""))
                continue

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
        html = "<b>Input parameters (superset)</b><table class='res-table'>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        for name in union_order:
            html += f"<tr><td>{name}</td><td>{fmap[name]}</td></tr>"
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
    finally:
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

    # Layout: column 0 = feature names (row labels); columns 1.. = one sample each.
    # Features are matched to each model BY NAME, so the CSV may be a superset
    # and rows may be in any order.
    csv_feat_names = [str(dataDf.iloc[r, 0]).strip() for r in range(dataDf.shape[0])]
    csv_feat_index = {name: r for r, name in enumerate(csv_feat_names)}

    sample_cols = list(range(1, dataDf.shape[1]))
    sample_names = [str(dataDf.columns[ci]) for ci in sample_cols]

    perf_keys = []
    # matrix[sample][perfkey] = (value_str, ood)
    matrix = {sn: {} for sn in sample_names}
    any_ood = False

    for (perfkey, folder) in selectors:
        perf_keys.append(perfkey)
        try:
            model_feats = await get_model_features(folder)
        except Exception as e:
            print(f"Error reading config for {perfkey} ({folder}): {e}")
            for sn in sample_names:
                matrix[sn][perfkey] = ("ERROR (config)", "")
            continue

        # Every feature this model needs must exist in the CSV
        missing = [f for f in model_feats if f not in csv_feat_index]
        if missing:
            msg = "MISSING: " + ", ".join(missing)
            for sn in sample_names:
                matrix[sn][perfkey] = (msg, "")
            continue

        try:
            dP, df, norm, ad_model = await load_model_bundle(folder)
        except Exception as e:
            print(f"Error loading {perfkey} ({folder}): {e}")
            for sn in sample_names:
                matrix[sn][perfkey] = ("ERROR", "")
            continue

        rows_for_model = [csv_feat_index[f] for f in model_feats]
        for ci, sn in zip(sample_cols, sample_names):
            try:
                col = dataDf.iloc[:, ci].tolist()
                R = np.array([[float(col[r]) for r in rows_for_model]])
                val, ood = run_prediction(dP, df, norm, ad_model, R)
                if ood:
                    any_ood = True
                matrix[sn][perfkey] = (f"{val:.5f}", ood)
            except (ValueError, TypeError):
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
