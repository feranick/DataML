#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML_BatchMaker
* v2025.09.24.1
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import pandas as pd
from io import BytesIO
import sys, io, csv, os
from js import document, Blob, URL
from pyscript import fetch, document

def rescaleList(list, value):
    list = [x + value for x in list]
    return list

async def batchCSV(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    numHeadColumns = document.querySelector("#numHeadColumns").value
    numHeadRows = document.querySelector("#numHeadRows").value
    charCCols = document.querySelector("#charCCols").value
    sampleLabelCol = document.querySelector("#sampleLabelCol").value
    validRows = document.querySelector("#validRows").value

    print(f"Number of Head Columns to skip: {numHeadColumns}")
    print(f"Number of Head Rows to skip: {numHeadRows}")
    print(f"List of Parameters/Features:  {charCCols}")
    print(f"Column with label of samples: {sampleLabelCol}")
    print(f"Rows with samples to extract:  {validRows}")

    output = '_________________________________________'
    output += '\n\nNumber of Head Columns to skip: '+numHeadColumns
    output += '\n\nNumber of Head Rows to skip: '+numHeadRows
    output += '\n\nList of Parameters/Features: '+charCCols
    output += '\n\nColumn with label of samples: '+sampleLabelCol
    output += '\n\nRows with samples to extract: '+validRows
    output_div.innerText = output
    
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
    
    validRowsList = [int(item) for item in validRows.split(',')]
    charCColsList = [int(item) for item in charCCols.split(',')]
    
    charCColsList = rescaleList(charCColsList, int(numHeadColumns) - 1)
    validRowsList = rescaleList(validRowsList, int(numHeadRows)-1)

    print(f"Valid rows: {validRowsList}")
    df = dataDf.iloc[validRowsList]
    
    paramNames = df.columns[charCColsList].tolist()
    sampleNames = df[df.columns[0]].values
    
    batchFile = os.path.splitext(inputFile.name)[0]+"_"+str(len(paramNames))+"params.csv"
    
    print(f"Parameter Names: {paramNames}")
    print(f"Sample Names: {sampleNames}")
    
    df2 = df[df.columns[charCColsList]].T
    df2.columns = sampleNames
    print(f"New batch matrix:\n{df2}")
    print(f"Batch CSV save in: {batchFile}")

    await create_csv_download(df2,None,batchFile)
    
async def create_csv_download(df, headers=None, filename="results.csv"):
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
    
    if df is None or df.size == 0:
        print("Input NumPy array is empty or None.")
        # Display the error message INSIDE the verified download_div
        download_div.innerText = "Error: No data to process for download."
        return
    
    try:
        # Use io.StringIO to act like an in-memory file
        output = io.StringIO()
        df.to_csv(output)
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
