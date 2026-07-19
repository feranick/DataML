#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Multi-Perf Regressor
* Offline tkinter UI
* version: 2026.07.19.2
* Uses: tkinter
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''
import os
import re
import csv
import json
import pickle
import configparser
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox

# Optional drag-and-drop support. Requires the 'tkinterdnd2' package
# (pip install tkinterdnd2). If unavailable, the CSV button still works;
# only the drag-and-drop convenience is disabled.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

try:
    from libDataML import *
except ImportError:
    messagebox.showerror("Import Error",
                         "Could not import libDataML. Ensure ui.py is in the same directory as libDataML.py")

STATE_FILE = ".ui_multiperf_state.json"
MODELS_LIST_FILE = "models_list.json"


def get_perf_key(folder):
    """Rule (b): match Perf<number><optional uppercase letters>, e.g. Perf1, Perf1B."""
    m = re.match(r'^(Perf\d+[A-Z]*)', folder)
    return m.group(1) if m else None


def perf_sort_key(key):
    m = re.match(r'^Perf(\d+)([A-Z]*)$', key)
    return (int(m.group(1)), m.group(2)) if m else (9999, key)


class ModelConfig:
    """Loads the DataML_DF.ini configuration for a specific model folder."""
    def __init__(self, folder):
        self.folder = folder
        self.ini_path = os.path.join(folder, "DataML_DF.ini")
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str

        if os.path.exists(self.ini_path):
            self.conf.read(self.ini_path)
            self.typeDF = self.conf.get('Parameters', 'typeDF', fallback='GradientBoosting')
            self.regressor = self.conf.getboolean('Parameters', 'regressor', fallback=True)
            self.normalize = self.conf.getboolean('Parameters', 'normalize', fallback=False)
        else:
            self.typeDF = 'GradientBoosting'
            self.regressor = True
            self.normalize = False

        self.mode = "Regressor" if self.regressor else "Classifier"
        self.modelName = os.path.join(folder, f"model_DF_{self.typeDF}{self.mode}.pkl")
        self.norm_file = os.path.join(folder, "norm_file.pkl")
        self.model_ad = os.path.join(folder, "model_ad.pkl")


class DataMLMultiApp:
    def __init__(self, root):
        self.root = root
        self.app_title = "DataML Multi-Perf - Offline"
        self.perf_groups = {}     # key -> [folders]
        self.sorted_keys = []
        self.features = []
        self.entry_widgets = {}
        self.perf_combos = {}     # key -> Combobox
        self.params_visible = False   # input parameters start collapsed
        self.state = self.load_state()

        self.load_model_list()
        self.root.title(self.app_title)
        self.setup_ui()

        if self.sorted_keys:
            self.build_feature_entries()

    # ---------- state persistence ----------
    def load_state(self):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(self):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f)
        except Exception as e:
            print(f"Could not save state: {e}")

    # ---------- directory scan / grouping ----------
    def load_model_list(self):
        """
        Build the Perf groups. Prefer models_list.json (same source as the web app);
        fall back to directory scanning if the JSON is missing or unreadable.
        """
        folders = None

        # 1. Try models_list.json first (keeps offline + web app in sync)
        if os.path.exists(MODELS_LIST_FILE):
            try:
                with open(MODELS_LIST_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    folders = data
                else:
                    print(f"'{MODELS_LIST_FILE}' is empty or not a list; falling back to directory scan.")
            except Exception as e:
                print(f"Could not read '{MODELS_LIST_FILE}': {e}; falling back to directory scan.")

        # 2. Fall back to scanning the working directory
        if folders is None:
            folders = self.scan_model_directories()

        # 3. Group folders by Perf key (rule b)
        for item in sorted(folders):
            key = get_perf_key(item)
            if key is None:
                print(f"Skipping folder (no Perf key): {item}")
                continue
            self.perf_groups.setdefault(key, []).append(item)

        for k in self.perf_groups:
            self.perf_groups[k].sort()
        self.sorted_keys = sorted(self.perf_groups.keys(), key=perf_sort_key)

        if not self.sorted_keys:
            messagebox.showwarning("Warning", "No valid Perf model subfolders found.")

    def scan_model_directories(self):
        """Fallback: scan the working directory for valid model subfolders. Returns a list of folder names."""
        current_dir = os.getcwd()
        folders = []
        for item in sorted(os.listdir(current_dir)):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item != '__pycache__':
                if os.path.exists(os.path.join(item_path, 'config.txt')) or \
                   os.path.exists(os.path.join(item_path, 'DataML_DF.ini')):
                    folders.append(item)
        return folders

    # ---------- UI ----------
    def setup_ui(self):
        self.root.geometry("720x820")
        self.root.configure(padx=20, pady=20)

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Toggle.TLabel", foreground="#007bff", font=("Helvetica", 10, "bold"))

        # --- Perf model selectors (own titled box) ---
        sel_lf = ttk.LabelFrame(self.root, text="Select a model for each performance parameter", padding=12)
        sel_lf.pack(fill=tk.X, pady=(0, 12))

        for key in self.sorted_keys:
            row = ttk.Frame(sel_lf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=key + ":", width=10, anchor="w",
                      font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
            combo = ttk.Combobox(row, values=self.perf_groups[key], state="readonly", width=45)
            combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # restore previous selection
            saved = self.state.get(key)
            if saved in self.perf_groups[key]:
                combo.set(saved)
            else:
                combo.current(0)
            combo.bind("<<ComboboxSelected>>", lambda e, k=key: self.on_select(k))
            self.perf_combos[key] = combo

        # --- Input parameters (collapsible) ---
        self.manual_lf = ttk.LabelFrame(self.root, text="Input parameters", padding=12)
        self.manual_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        # Collapsible toggle header (mirrors the web "Show input parameters" triangle)
        self.toggle_lbl = ttk.Label(self.manual_lf, text="▶ Show input parameters",
                                    style="Toggle.TLabel", cursor="hand2")
        self.toggle_lbl.pack(anchor="w", pady=(0, 8))
        self.toggle_lbl.bind("<Button-1>", lambda e: self.toggle_params())

        # Container wrapping canvas + scrollbar so the whole block can be hidden
        self.params_container = ttk.Frame(self.manual_lf)
        # NOT packed initially -> parameters collapsed by default

        self.canvas = tk.Canvas(self.params_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.params_container, orient="vertical", command=self.canvas.yview)
        self.features_frame = ttk.Frame(self.canvas)
        self.features_frame.bind("<Configure>",
                                 lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.features_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.predict_btn = ttk.Button(self.root, text="Predict all Perf", command=self.single_predict)
        self.predict_btn.pack(pady=(0, 12))

        # --- Batch CSV ---
        csv_lf = ttk.LabelFrame(self.root, text="Predict using CSV file", padding=12)
        csv_lf.pack(fill=tk.X, pady=(0, 12))
        self.csv_btn = ttk.Button(csv_lf, text="Select a CSV File to predict",
                                  command=self.batch_predict)
        self.csv_btn.pack()

        # Drag-and-drop hint + registration (only if tkinterdnd2 is available)
        if _HAS_DND:
            hint = ttk.Label(csv_lf, text="or drag & drop a CSV file here", foreground="#888")
            hint.pack(pady=(8, 0))
            for w in (csv_lf, self.csv_btn, hint):
                w.drop_target_register(DND_FILES)
                w.dnd_bind('<<Drop>>', self.on_drop_csv)

        # --- Output ---
        out_lf = ttk.LabelFrame(self.root, text="Output", padding=10)
        out_lf.pack(fill=tk.BOTH, expand=True)
        self.output_text = scrolledtext.ScrolledText(out_lf, wrap=tk.NONE, font=("Courier", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def toggle_params(self):
        """Show/hide the auto-generated input parameter fields."""
        self.params_visible = not self.params_visible
        if self.params_visible:
            self.params_container.pack(fill=tk.BOTH, expand=True)
            self.toggle_lbl.config(text="▼ Hide input parameters")
        else:
            self.params_container.pack_forget()
            self.toggle_lbl.config(text="▶ Show input parameters")

    def on_drop_csv(self, event):
        """Handle a file dropped onto the CSV box (requires tkinterdnd2)."""
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if not paths:
            return
        path = paths[0]
        if not path.lower().endswith(".csv"):
            self.log_message("Please drop a .csv file.", clear=True)
            return
        self.batch_predict(filepath=path)

    def on_select(self, key):
        self.state[key] = self.perf_combos[key].get()
        self.save_state()
        # features are shared, but rebuild in case the first group's config changes
        self.build_feature_entries()

    def get_first_folder(self):
        if not self.sorted_keys:
            return None
        return self.perf_combos[self.sorted_keys[0]].get()

    def build_feature_entries(self):
        folder = self.get_first_folder()
        if not folder:
            return
        config_path = os.path.join(folder, "config.txt")
        self.features = []
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.features = [feat.strip() for feat in content.split(",")]

        for widget in self.features_frame.winfo_children():
            widget.destroy()
        self.entry_widgets = {}

        for i, feat in enumerate(self.features):
            row_frame = ttk.Frame(self.features_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=feat, width=24, anchor="w").pack(side=tk.LEFT)
            ent = ttk.Entry(row_frame)
            ent.insert(0, str(i))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry_widgets[feat] = ent

    def log_message(self, message, clear=False):
        if clear:
            self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    # ---------- model loading / prediction ----------
    def load_bundle(self, folder):
        cfg = ModelConfig(folder)
        with open(cfg.modelName, 'rb') as f:
            df = pickle.load(f)
        norm = None
        if cfg.normalize:
            try:
                with open(cfg.norm_file, 'rb') as f:
                    norm = pickle.load(f)
            except Exception as e:
                print(f"Normalizer load error: {e}")
        ad_model = None
        try:
            with open(cfg.model_ad, 'rb') as f:
                ad_model = pickle.load(f)
        except Exception:
            pass
        return cfg, df, norm, ad_model

    def run_prediction(self, cfg, df, norm, ad_model, R):
        Rp = np.copy(R)
        if cfg.normalize and norm is not None:
            Rp = norm.transform_valid_data(Rp)
        ood = ""
        if ad_model is not None:
            try:
                flags = ad_model.predict(Rp)
                if flags[0] == -1:
                    ood = "[OOD]"
            except ValueError:
                pass
        if cfg.normalize and norm is not None:
            pred = norm.transform_inverse_single(df.predict(Rp))
        else:
            pred = df.predict(Rp)
        return float(pred[0]), ood

    # ---------- single predict ----------
    def single_predict(self):
        if not self.sorted_keys:
            return
        self.log_message("Please wait...", clear=True)
        ok, report = self.verify_feature_consistency()
        if not ok:
            self.log_message(report, clear=True)
            return

        try:
            R_list = [float(self.entry_widgets[feat].get()) for feat in self.features]
        except ValueError:
            self.log_message("Error: all input parameters must be valid numbers.", clear=True)
            return
        R = np.array([R_list])
        Rorig = np.copy(R)

        results = []
        any_ood = False
        for key in self.sorted_keys:
            folder = self.perf_combos[key].get()
            try:
                cfg, df, norm, ad_model = self.load_bundle(folder)
                val, ood = self.run_prediction(cfg, df, norm, ad_model, R)
                if ood:
                    any_ood = True
                results.append((key, folder, f"{val:.5f}", ood))
            except Exception as e:
                results.append((key, folder, "ERROR", ""))
                print(f"Error predicting {key} ({folder}): {e}")

        out = "Input parameters\n"
        out += "----------------------------------------\n"
        for i, feat in enumerate(self.features):
            out += f"  {feat} = {Rorig[0][i]}\n"
        out += "\nPredicted performance parameters\n"
        out += "----------------------------------------------------------\n"
        out += f"{'Perf':<10}{'Predicted value':<18}{'AD':<8}  Model\n"
        out += "----------------------------------------------------------\n"
        for (key, folder, val, ood) in results:
            out += f"{key:<10}{val:<18}{ood:<8}  {folder}\n"
        if any_ood:
            out += ("\nWARNING: [OOD] parameters fall OUTSIDE the known\n"
                    "Applicability Domain. Those predictions may be unreliable.\n")
        self.log_message(out, clear=True)

    # ---------- batch predict ----------
    def batch_predict(self, filepath=None):
        """`filepath` may be supplied by a drag-and-drop event; when None, an
        Open dialog is shown as before."""
        if not self.sorted_keys:
            return
        if filepath is None:
            filepath = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV Files", "*.csv")])
        if not filepath:
            return
        filename = os.path.basename(filepath)
        self.log_message("Processing CSV... Please wait.", clear=True)

        ok, report = self.verify_feature_consistency()
        if not ok:
            self.log_message(report, clear=True)
            return

        try:
            dataDf = pd.read_csv(filepath)
        except Exception as e:
            self.log_message(f"Error reading CSV '{filename}': {e}", clear=True)
            return

        if len(self.features) != dataDf.shape[0]:
            self.log_message("Please choose the right models for this file (feature count mismatch).", clear=True)
            return

        sample_names = [str(dataDf.columns[ci]) for ci in range(1, dataDf.shape[1])]
        perf_keys = []
        matrix = {sn: {} for sn in sample_names}
        any_ood = False

        for key in self.sorted_keys:
            perf_keys.append(key)
            folder = self.perf_combos[key].get()
            try:
                cfg, df, norm, ad_model = self.load_bundle(folder)
            except Exception as e:
                print(f"Error loading {key} ({folder}): {e}")
                for sn in sample_names:
                    matrix[sn][key] = ("ERROR", "")
                continue

            for ci in range(1, dataDf.shape[1]):
                sn = str(dataDf.columns[ci])
                try:
                    R = np.array([dataDf.iloc[:, ci].tolist()], dtype=float)
                    val, ood = self.run_prediction(cfg, df, norm, ad_model, R)
                    if ood:
                        any_ood = True
                    matrix[sn][key] = (f"{val:.5f}", ood)
                except ValueError:
                    matrix[sn][key] = ("ERROR", "")

        # On-screen table
        colw = 16
        out = f"Batch prediction for {filename}\n"
        out += "=" * (12 + colw * len(perf_keys)) + "\n"
        out += f"{'Sample':<12}" + "".join(f"{pk:<{colw}}" for pk in perf_keys) + "\n"
        out += "-" * (12 + colw * len(perf_keys)) + "\n"
        for sn in sample_names:
            row = f"{sn:<12}"
            for pk in perf_keys:
                val, ood = matrix[sn].get(pk, ("", ""))
                cell = val + (" " + ood if ood else "")
                row += f"{cell:<{colw}}"
            out += row + "\n"
        if any_ood:
            out += ("\nWARNING: [OOD] entries fall OUTSIDE the known\n"
                    "Applicability Domain. Those predictions may be unreliable.\n")
        self.log_message(out, clear=True)

        # CSV export
        summaryFile = [['File:', filename],
                       ['DataML_DF', 'Multi-Perf'],
                       [],
                       ['Sample'] + perf_keys]
        for sn in sample_names:
            row = [sn]
            for pk in perf_keys:
                val, ood = matrix[sn].get(pk, ("", ""))
                row.append(val + (" " + ood if ood else ""))
            summaryFile.append(row)

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"Results_{filename}",
            title="Save Predictions As",
            filetypes=[("CSV Files", "*.csv")])
        if save_path:
            try:
                with open(save_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(summaryFile)
                self.log_message(f"\n[Success] Results saved to:\n{save_path}")
            except Exception as e:
                self.log_message(f"\n[Error] Could not save results: {e}")

    def verify_feature_consistency(self):
        """
        Verify every selected model's config.txt matches the reference (self.features).
        Returns (ok: bool, report: str).
        """
        mismatches = []
        for key in self.sorted_keys:
            folder = self.perf_combos[key].get()
            config_path = os.path.join(folder, "config.txt")
            feats = []
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        feats = [feat.strip() for feat in content.split(",")]
            else:
                mismatches.append(f"  - {key} ({folder}): config.txt not found")
                continue

            if feats != self.features:
                if len(feats) != len(self.features):
                    detail = f"count {len(feats)} vs expected {len(self.features)}"
                else:
                    diffs = [f"[{i}] '{feats[i]}' != '{self.features[i]}'"
                             for i in range(len(feats)) if feats[i] != self.features[i]]
                    detail = "; ".join(diffs)
                mismatches.append(f"  - {key} ({folder}): {detail}")

        if mismatches:
            report = ("ERROR: Input parameter mismatch between models.\n"
                      "All Perf models must share identical config.txt features.\n\n"
                      "Reference (from first model):\n  " + ", ".join(self.features) +
                      "\n\nMismatched models:\n" + "\n".join(mismatches))
            return False, report
        return True, ""


if __name__ == "__main__":
    # Use the DnD-enabled Tk root when tkinterdnd2 is installed
    root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
    app = DataMLMultiApp(root)
    root.mainloop()
