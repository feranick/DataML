#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Classifier and Regressor
* Offline tkinter UI
* version: 2026.04.11.1
* Uses: tkinter
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''
import os
import re
import csv
import pickle
import configparser
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# Import the custom library natively. 
# This is required so pickle can un-serialize the Normalizer and MultiClassReductor.
try:
    from libDataML import *
except ImportError:
    messagebox.showerror("Import Error", "Could not import libDataML. Ensure offline_gui.py is in the same directory as libDataML.py")

class ModelConfig:
    """Helper class to load the DataML_DF.ini configuration for a specific model."""
    def __init__(self, folder):
        self.folder = folder
        self.ini_path = os.path.join(folder, "DataML_DF.ini")
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        
        if os.path.exists(self.ini_path):
            self.conf.read(self.ini_path)
            self.typeDF = self.conf.get('Parameters', 'typeDF', fallback='GradientBoosting')
            self.regressor = self.conf.getboolean('Parameters', 'regressor', fallback=False)
            self.normalize = self.conf.getboolean('Parameters', 'normalize', fallback=False)
        else:
            # Safe fallbacks if ini is missing
            self.typeDF = 'GradientBoosting'
            self.regressor = False
            self.normalize = False

        self.mode = "Regressor" if self.regressor else "Classifier"
        self.modelName = os.path.join(folder, f"model_DF_{self.typeDF}{self.mode}.pkl")
        self.model_le = os.path.join(folder, "model_le.pkl")
        self.norm_file = os.path.join(folder, "norm_file.pkl")
        self.model_ad = os.path.join(folder, "model_ad.pkl")

class DataMLOfflineApp:
    def __init__(self, root):
        self.root = root
        self.models = []
        self.app_title = "DataML - Offline"
        self.features = []
        self.entry_widgets = {}

        # Get title from HTML if possible, but build models list from directories
        self.get_app_title('index.html')
        self.scan_model_directories()
        
        self.root.title(self.app_title)
        
        self.setup_ui()
        if self.models:
            self.model_combo.current(0)
            self.on_model_select(None)

    def get_app_title(self, filepath):
        """Optionally extracts the title from index.html if it exists."""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
            if title_match:
                self.app_title = title_match.group(1).replace("pyscript version", "Offline Desktop Version").strip()

    def scan_model_directories(self):
        """Scans the current directory for subfolders that represent valid ML models."""
        current_dir = os.getcwd()
        subfolders = []

        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            
            # Check if it's a directory and not a hidden/system folder
            if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
                # Verify it's an actual model folder by checking for required files
                if os.path.exists(os.path.join(item_path, 'config.txt')) or os.path.exists(os.path.join(item_path, 'DataML_DF.ini')):
                    subfolders.append(item)

        # Sort alphabetically so the dropdown is organized
        self.models = sorted(subfolders)

        if not self.models:
            messagebox.showwarning("Warning", "No model subfolders found in the current directory.")

    def setup_ui(self):
        """Sets up the Tkinter GUI layout."""
        self.root.geometry("650x750")
        self.root.configure(padx=20, pady=20)

        # Style configurations
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 10))

        # --- Top Section: Model Selection ---
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(top_frame, text="Model:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Use self.models populated by directory scanning
        self.model_combo = ttk.Combobox(top_frame, values=self.models, state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_select)

        # --- Middle Section: Dynamic Features & Manual Predict ---
        self.manual_lf = ttk.LabelFrame(self.root, text="Manual Prediction", padding=15)
        self.manual_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Canvas/Scrollbar for dynamic features in case there are many
        self.canvas = tk.Canvas(self.manual_lf, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.manual_lf, orient="vertical", command=self.canvas.yview)
        self.features_frame = ttk.Frame(self.canvas)

        self.features_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.features_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.predict_btn = ttk.Button(self.root, text="Predict", command=self.single_predict)
        self.predict_btn.pack(pady=(0, 15))

        # --- Middle Section: Batch CSV Predict ---
        csv_lf = ttk.LabelFrame(self.root, text="Predict using CSV file", padding=15)
        csv_lf.pack(fill=tk.X, pady=(0, 15))

        self.csv_btn = ttk.Button(csv_lf, text="Select a CSV File to predict", command=self.batch_predict)
        self.csv_btn.pack()

        # --- Bottom Section: Output Log ---
        out_lf = ttk.LabelFrame(self.root, text="Output", padding=10)
        out_lf.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(out_lf, wrap=tk.WORD, font=("Courier", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def on_model_select(self, event):
        """Triggers when a new model is selected. Loads its specific config.txt to generate entries."""
        folder = self.model_combo.get()
        if not folder: return

        config_path = os.path.join(folder, "config.txt")
        self.features = []
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.features = [feat.strip() for feat in content.split(",")]

        # Shorten button label if folder name is very long
        display_name = folder[:15] + "..." if len(folder) > 15 else folder
        self.predict_btn.config(text=f"Predict {display_name}")
        
        self.build_feature_entries()
        self.log_message(f"Loaded configuration for model: {folder}")

    def build_feature_entries(self):
        """Dynamically clears and recreates the input fields based on the selected model's features."""
        for widget in self.features_frame.winfo_children():
            widget.destroy()

        self.entry_widgets = {}
        for i, feat in enumerate(self.features):
            row_frame = ttk.Frame(self.features_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(row_frame, text=feat, width=20, anchor="w")
            lbl.pack(side=tk.LEFT)
            
            ent = ttk.Entry(row_frame)
            ent.insert(0, str(i)) # Setting default value just like PyScript logic fv[i] = i
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry_widgets[feat] = ent

    def log_message(self, message, clear=False):
        """Helper to write to the ScrolledText output."""
        if clear:
            self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def single_predict(self):
        """Replicates singlePredict logic from PyScript."""
        folder = self.model_combo.get()
        if not folder: return
        self.log_message("Please wait...", clear=True)
        
        cfg = ModelConfig(folder)
        
        # 1. Gather inputs
        try:
            R_list = [float(self.entry_widgets[feat].get()) for feat in self.features]
        except ValueError:
            self.log_message("Error: All feature entries must be valid numbers.", clear=True)
            return

        R = np.array([R_list])
        Rorig = np.copy(R)

        # 2. Load Normalizer (if applicable)
        norm = None
        if cfg.normalize:
            try:
                with open(cfg.norm_file, 'rb') as f:
                    norm = pickle.load(f)
                R = norm.transform_valid_data(R)
            except Exception as e:
                self.log_message(f"Normalizer load error: {e}")

        # 3. Load AD Model (if applicable)
        ad_tag = ""
        try:
            with open(cfg.model_ad, 'rb') as f:
                ad_model = pickle.load(f)
            safety_flags = ad_model.predict(R)
            for flag in safety_flags:
                if flag == -1:
                    ad_tag = "\nWARNING: Sample features fall \nOUTSIDE the known Applicability Domain! \nPrediction may be unreliable."
        except:
            pass # AD model missing is fine

        # 4. Load Engine & Predict
        try:
            with open(cfg.modelName, 'rb') as f:
                df = pickle.load(f)
        except Exception as e:
            self.log_message(f"Error loading model '{cfg.modelName}': {e}", clear=True)
            return

        output = f"============================\n"

        if cfg.regressor:
            if cfg.normalize and norm:
                pred = norm.transform_inverse_single(df.predict(R))
            else:
                pred = df.predict(R)
            output += f" {folder[:15]} = {pred[0]:.5f}\n"
        else:
            with open(cfg.model_le, 'rb') as f:
                le = pickle.load(f)
            pred = le.inverse_transform_bulk(df.predict(R))
            pred_classes = le.inverse_transform_bulk(df.classes_)
            proba = df.predict_proba(R)
            ind = np.where(proba[0] == np.max(proba[0]))[0]

            output += ' Prediction\t| Probability [%]\n'
            output += '------------------------------------\n'
            for j in range(len(ind)):
                if cfg.normalize and norm:
                    p_class = round(norm.transform_inverse_single(pred_classes[ind[j]]), 2)
                else:
                    p_class = round(pred_classes[ind[j]], 2)
                output += f" {p_class}\t\t|  {str(100 * proba[0][ind[j]])[:5]}\n"

        # 5. Format Output Log
        output += '============================\n'
        for i in range(len(self.features)):
            output += f" {self.features[i]} = {Rorig[0][i]}\n"
        output += '============================\n'
        output += f"{cfg.typeDF} {cfg.mode}\n"
        output += '============================\n'
        output += ad_tag

        self.log_message(output, clear=True)

    def batch_predict(self):
        """Replicates batchPredict logic from PyScript."""
        filepath = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV Files", "*.csv")])
        if not filepath: return
        
        folder = self.model_combo.get()
        cfg = ModelConfig(folder)
        filename = os.path.basename(filepath)
        self.log_message("Processing CSV... Please wait.", clear=True)

        try:
            dataDf = pd.read_csv(filepath)
        except Exception as e:
            self.log_message(f"Error reading CSV '{filename}': {e}", clear=True)
            return

        if len(self.features) != dataDf.shape[0]:
            self.log_message("Please choose the right model for this file (Feature count mismatch).", clear=True)
            return

        # Setup Normalizer & AD Model
        norm, ad_model = None, None
        if cfg.normalize:
            try:
                with open(cfg.norm_file, 'rb') as f:
                    norm = pickle.load(f)
            except Exception as e:
                self.log_message(f"Normalizer error: {e}")

        try:
            with open(cfg.model_ad, 'rb') as f:
                ad_model = pickle.load(f)
        except:
            pass

        # Load Core Model
        try:
            with open(cfg.modelName, 'rb') as f:
                df = pickle.load(f)
        except Exception as e:
            self.log_message(f"Error loading model '{cfg.modelName}': {e}", clear=True)
            return

        # CSV Export Setup
        summaryFile = [['File:', filename, ''], ['DataML_DF', cfg.typeDF, cfg.mode], ['Model', folder, '']]
        le = None
        if cfg.regressor:
            summaryFile.append(['Sample', 'Predicted Value', ''])
        else:
            with open(cfg.model_le, 'rb') as f:
                le = pickle.load(f)
            summaryFile.append(['Sample', 'Predicted Value', 'Probability %'])

        output = f"======================================\n Prediction for {folder[:15]}\n======================================\n"
        overall_ad_tag = ""

        for i in range(1, dataDf.shape[1]):
            R = np.array([dataDf.iloc[:, i].tolist()], dtype=float)
            col_name = dataDf.columns[i]
            
            if cfg.normalize and norm:
                R = norm.transform_valid_data(R)

            ood_tag = ""
            if ad_model is not None:
                safety_flags = ad_model.predict(R)
                if safety_flags[0] == -1:
                    overall_ad_tag = "\nWARNING: Sample features fall \nOUTSIDE the known Applicability Domain! \nPrediction may be unreliable."
                    ood_tag = "[OOD]"

            if cfg.regressor:
                if cfg.normalize and norm:
                    pred = norm.transform_inverse_single(df.predict(R))
                else:
                    pred = df.predict(R)
                
                output += f" {col_name} = {pred[0]:.5f} \t {ood_tag}\n"
                summaryFile.append([col_name, pred[0], ''])
            else:
                pred = le.inverse_transform_bulk(df.predict(R))
                pred_classes = le.inverse_transform_bulk(df.classes_)
                proba = df.predict_proba(R)
                ind = np.where(proba[0] == np.max(proba[0]))[0]

                for j in range(len(ind)):
                    if cfg.normalize and norm:
                        p_class = str(round(norm.transform_inverse_single(pred_classes[ind[j]]), 2))
                    else:
                        p_class = str(round(pred_classes[ind[j]], 2))
                    
                    output += f" {col_name} = {p_class}\t\t|  {str(100 * proba[0][ind[j]])[:5]}%) {ood_tag}\n"
                    summaryFile.append([col_name, pred_classes[ind[j]], round(100 * proba[0][ind[j]], 1)])

        output += f"=======================================\n{cfg.typeDF} {cfg.mode}\n=======================================\n{overall_ad_tag}"
        self.log_message(output, clear=True)

        # Output file saving logic
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", 
            initialfile=f"Results_{filename}",
            title="Save Predictions As",
            filetypes=[("CSV Files", "*.csv")]
        )
        
        if save_path:
            try:
                with open(save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(summaryFile)
                self.log_message(f"\n[Success] Results saved to:\n{save_path}")
            except Exception as e:
                self.log_message(f"\n[Error] Could not save results: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataMLOfflineApp(root)
    root.mainloop()
