#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
******************************************************************
* Add individual Perf sheets into CorrAnalysis Result master XLSX
* v2025.10.16.1
* Uses: Pandas
* By: Nicola Ferralis <feranick@hotmail.com>
******************************************************************
'''
print(__doc__)
import sys, os.path, shutil
import pandas as pd

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py AddSinglePerfCorrAnalMaster.py <file in xslx format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(f" Opening CorrAnalsyis Result Master File: {sys.argv[1]}")
        AddPerfTabs(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def AddPerfTabs(xlsx):
    newXlsx = os.path.splitext(xlsx)[0]+"_with-PERF.xlsx"
    try:
        shutil.copy(xlsx, newXlsx)
        print(f" File copied successfully from '{xlsx}' to '{newXlsx}'")
    except FileNotFoundError:
        print(f" Error: The source file '{xlsx}' was not found.")
    except Exception as e:
        print(f" An error occurred during copy: {e}")
    
    df_orig = pd.read_excel(xlsx, sheet_name='Complete')
    df = df_orig.dropna(subset=['PERF'])
    print(df)
    perf = df['PERF'].unique().tolist()
    
    for p in perf:
        print(f"\n Processing parameter: {p}")
        df_sel = df[df['PERF'] == p]
        dfc = df_sel.sort_values(by=df_sel.columns[0], ascending=True)
        
        duplicate_columns = ['PAR', 'PERF']
        priority_column = 'Corr'
        temp_abs_col = '_ABS_PRIORITY_COL'
        
        dfc[temp_abs_col] = dfc[priority_column].abs()
        
        df_sorted = dfc.sort_values(
            by=priority_column,
            ascending=False
            )
        df_no_duplicates = df_sorted.drop_duplicates(
            subset=duplicate_columns,
            keep='first'
            )
        
        df_drop = df_no_duplicates.drop(columns=[temp_abs_col])
        df_final = df_drop.sort_values(by=df_drop.columns[0], ascending=True)
        #print(new_df)
        par_label = "_".join(p.split()[-2:])
        
        with pd.ExcelWriter(newXlsx, mode='a', engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name=par_label, index=False)
        
        print(f" Added new tab {par_label} to {newXlsx}")
    
    print("\n")
        
        
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


