#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*******************************************************
* Split CorrAnalysis master into individual Perf files
* v2025.10.16.1
* Uses: Pandas
* By: Nicola Ferralis <feranick@hotmail.com>
*******************************************************
'''
print(__doc__)
import sys, os.path
import pandas as pd

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py splitPerfFromCorrAnal <file in csv format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Creating from',sys.argv[1],'into CSV for each Perf...')
        createCSVs(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def createCSVs(csv):
    df_orig = pd.read_csv(csv)
    df = df_orig.dropna(subset=['PERF'])
    print(df)
    perf = df['PERF'].unique().tolist()
    print(perf)
    
    for p in perf:
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
        new_ext = "_".join(p.split()[-2:])
        perfFile = os.path.splitext(csv)[0]+"_"+new_ext+".csv"
        df_final.to_csv(perfFile, index=False)
        
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


