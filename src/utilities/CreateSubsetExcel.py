#!/usr/bin/env python3
'''
**************************************************
* Create Subset based on sample/material code 
* from provided xlsx
* version: v2025.11.21.2
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************
'''
print(__doc__)
import pandas as pd
import sys, os.path, h5py

#************************************
# Parameters definition
#************************************
class Conf:
    sheet_name = "analysis"
    rows_to_skip = 4
    row_with_name = 3
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 4:
        print(' Usage:\n  python3 CreateSubsetExcel.py <Excel File> <col> <code>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    conf = Conf()
    
    print(" Opening Excel File:",sys.argv[1],"...\n")
    
    xls = pd.ExcelFile(sys.argv[1])
    
    if conf.sheet_name in xls.sheet_names:
        sheet_name = conf.sheet_name
        print(f" Sheet name '{conf.sheet_name}' found.\n")
    else:
        sheet_name = xls.sheet_names[0]
        print(f" Sheet name '{conf.sheet_name}' NOT found, first sheet in file.\n")
    
    df_orig = pd.read_excel(sys.argv[1], sheet_name=sheet_name, header=None)
    
    rootFile = os.path.splitext(sys.argv[1])[0]
    column_name = df_orig.iloc[3,int(sys.argv[2])].replace(' ','-')
    newFile = rootFile + "_" + column_name + "_" + sys.argv[3] + ".csv"
    
    print(" Original table:\n", df_orig)
    
    df_result = create_subset_df(df_orig, int(sys.argv[2]), sys.argv[3])
    
    print(df_result)
    
    df_result.to_csv(newFile, index=False, header=False)
    print(f"\n Table saved in: {newFile}\n")
    
#******************************************************************
# create Dataframe with only data with specific condition
#******************************************************************
def create_subset_df(df, col, string):
    conf = Conf()
    
    mask = df[df.columns[col]] == string
    mask.iloc[:conf.rows_to_skip] = True
    df_new = df[mask]
    return df_new

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
