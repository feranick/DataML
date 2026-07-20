#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* DataML Decision Forests - Multi-Perf Regressor
* Offline tkinter UI - SUPERSET (name-matched features)
* version: 2026.07.20.3
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


def read_model_features(folder):
    """
    Return this model's feature-name list (in the order the model expects),
    or None if config.txt is missing. An empty file yields an empty list.
    """
    config_path = os.path.join(folder, "config.txt")
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []
    return [s.strip() for s in content.split(",") if s.strip()]


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
        self.features = []        # the SUPERSET (union) of all model features
        self.entry_widgets = {}   # feature_name -> Entry widget
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

        # --- Input parameters (superset of all models, collapsible) ---
        self.manual_lf = ttk.LabelFrame(self.root, text="Input parameters (superset of all models)", padding=12)
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
        # Persist the selection only. The input field set is the UNION over ALL
        # models, so it does not depend on the current selection and is NOT rebuilt
        # here — rebuilding would discard values the user has already typed.
        self.state[key] = self.perf_combos[key].get()
        self.save_state()

    def compute_feature_union(self):
        """
        Union (superset) of every model's config.txt features, preserving
        first-appearance order over the folders (already sorted) for a stable,
        deterministic layout.
        """
        all_folders = []
        for key in self.sorted_keys:
            all_folders.extend(self.perf_groups[key])

        union, seen = [], set()
        for folder in all_folders:
            feats = read_model_features(folder)
            if feats is None:
                print(f"config.txt not found for '{folder}'; skipping its features.")
                continue
            for f in feats:
                if f not in seen:
                    seen.add(f)
                    union.append(f)
        return union

    def build_feature_entries(self):
        # Field set = UNION (superset) of every model's config.txt features.
        # Selection-independent, so this is built once at startup.
        self.features = self.compute_feature_union()

        for widget in self.features_frame.winfo_children():
            widget.destroy()
        self.entry_widgets = {}

        if not self.features:
            ttk.Label(self.features_frame,
                      text="No features found in any model config.txt.",
                      anchor="w").pack(fill=tk.X, pady=2)
            return

        for i, feat in enumerate(self.features):
            row_frame = ttk.Frame(self.features_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=feat, width=24, anchor="w").pack(side=tk.LEFT)
            ent = ttk.Entry(row_frame)
            ent.insert(0, str(i))
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry_widgets[feat] = ent

    def get_feature_map(self):
        """Return {feature_name: raw_string_value} from the superset input fields."""
        return {feat: w.get() for feat, w in self.entry_widgets.items()}

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

        # Shared superset of inputs (name -> value)
        fmap = self.get_feature_map()
        if not fmap:
            self.log_message("No input fields available.", clear=True)
            return

        results = []
        any_ood = False
        for key in self.sorted_keys:
            folder = self.perf_combos[key].get()

            model_feats = read_model_features(folder)
            if model_feats is None:
                results.append((key, folder, "ERROR (no config.txt)", ""))
                continue

            # Pull just this model's features, by name, from the superset
            R, err = build_R_from_map(model_feats, fmap)
            if err:
                results.append((key, folder, "ERROR: " + err, ""))
                continue

            try:
                cfg, df, norm, ad_model = self.load_bundle(folder)
                val, ood = self.run_prediction(cfg, df, norm, ad_model, R)
                if ood:
                    any_ood = True
                results.append((key, folder, f"{val:.5f}", ood))
            except Exception as e:
                results.append((key, folder, "ERROR", ""))
                print(f"Error predicting {key} ({folder}): {e}")

        out = "Input parameters (superset)\n"
        out += "----------------------------------------\n"
        for feat in self.features:
            out += f"  {feat} = {fmap[feat]}\n"
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

        try:
            dataDf = pd.read_csv(filepath)
        except Exception as e:
            self.log_message(f"Error reading CSV '{filename}': {e}", clear=True)
            return

        # Layout: column 0 = feature names (row labels); columns 1.. = one sample each.
        # Features are matched to each model BY NAME, so the CSV may be a superset
        # and rows may be in any order.
        csv_feat_names = [str(dataDf.iloc[r, 0]).strip() for r in range(dataDf.shape[0])]
        csv_feat_index = {name: r for r, name in enumerate(csv_feat_names)}

        sample_cols = list(range(1, dataDf.shape[1]))
        sample_names = [str(dataDf.columns[ci]) for ci in sample_cols]
        perf_keys = []
        matrix = {sn: {} for sn in sample_names}
        any_ood = False

        for key in self.sorted_keys:
            perf_keys.append(key)
            folder = self.perf_combos[key].get()

            model_feats = read_model_features(folder)
            if model_feats is None:
                for sn in sample_names:
                    matrix[sn][key] = ("ERROR (no config.txt)", "")
                continue

            # Every feature this model needs must exist in the CSV
            missing = [f for f in model_feats if f not in csv_feat_index]
            if missing:
                msg = "MISSING: " + ", ".join(missing)
                for sn in sample_names:
                    matrix[sn][key] = (msg, "")
                continue

            try:
                cfg, df, norm, ad_model = self.load_bundle(folder)
            except Exception as e:
                print(f"Error loading {key} ({folder}): {e}")
                for sn in sample_names:
                    matrix[sn][key] = ("ERROR", "")
                continue

            rows_for_model = [csv_feat_index[f] for f in model_feats]
            for ci, sn in zip(sample_cols, sample_names):
                try:
                    col = dataDf.iloc[:, ci].tolist()
                    R = np.array([[float(col[r]) for r in rows_for_model]])
                    val, ood = self.run_prediction(cfg, df, norm, ad_model, R)
                    if ood:
                        any_ood = True
                    matrix[sn][key] = (f"{val:.5f}", ood)
                except (ValueError, TypeError):
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


if __name__ == "__main__":
    # Use the DnD-enabled Tk root when tkinterdnd2 is installed
    root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
    app = DataMLMultiApp(root)
    root.mainloop()
