#!/usr/bin/env python3
'''
**************************************************
* Create Master Dataset from provided xlsx
* 
* version: v2025.11.26.1
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
    sheet_name_original = "as_received"
    sheet_name_new = "analysis"
    row_with_name = 3
    replace_NaN = True
    value_for_NaN = 0
    skip_NaN_rows = True
    rows_to_skip = 3
    cols_to_skip = 2
    drop_cols_with_code = True
    autofind_cols_with_code = False
    manual_cols_with_code = [2,9]
    
    drop_first_column = True

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 CreateMasterDatasetExcel.py <Excel File>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    conf = Conf()
    
    print(" Opening Excel File:",sys.argv[1],"...\n")
    
    '''
    if conf.sheet_name_new in xls.sheet_names:
        sheet_name = conf.sheet_name_new
        print(f" Sheet name '{conf.sheet_name_new}' found.\n")
    elif conf.sheet_name_original in xls.sheet_names:
        sheet_name = conf.sheet_name_original
        print(f" Sheet name '{conf.sheet_name_new}' NOT found, using '{conf.sheet_name_original}'.\n")
    else:
        sheet_name = xls.sheet_names[0]
        print(f" Sheet name '{conf.sheet_name_new}' NOT found, first sheet in file.\n")
    '''
    
    '''
    xls = pd.ExcelFile(sys.argv[1])
    sheet_name = xls.sheet_names[0]
    print(sheet_name)
    xls.close()
    '''
    
    df_original = pd.read_excel(sys.argv[1], header=None)

    print(" Original table:\n", df_original)
    
    df = df_original.copy()
    
    print(df[df.columns[[2,3,4]]])

    formatted_cols_to_drop = find_formatted_columns(df)
    
    if conf.drop_cols_with_code:
        if conf.autofind_cols_with_code:
            print(f"\n Dropping columns with 'AA-BB' - autofind: {formatted_cols_to_drop}")
            df = df.drop(columns=df.columns[formatted_cols_to_drop], axis=1)
        else:
            print(f"\n Dropping columns with 'AA-BB' - manual: {conf.manual_cols_with_code}")
            df = df.drop(columns=df.columns[conf.manual_cols_with_code], axis=1)

    if conf.skip_NaN_rows:
        df = df.dropna(how='all', axis=0)
    
    if conf.replace_NaN:
        df_slice = df.iloc[conf.rows_to_skip:, conf.cols_to_skip:]
        df_slice = df_slice.fillna(conf.value_for_NaN).infer_objects(copy=False)
        df.iloc[conf.rows_to_skip:, conf.cols_to_skip:] = df_slice
            
    if conf.drop_first_column:
        print("\n Dropping first column")
        df = df.drop(df.columns[0], axis=1)
            
    print(f"\n Modified table: {df}\n",)
    
    with pd.ExcelWriter(sys.argv[1], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=conf.sheet_name_new, index=False, header=False)
    
    print(f"\n Table saved in: {sys.argv[1]}\n")

#************************************
# Identify columns with AA-BB codes
#************************************
def find_formatted_columns(df):
    # The regular expression pattern for "XX-YY" where X and Y are digits
    pattern = r"^\d{2}-\d{2}$"
    
    matching_columns = []
    
    for col in df.columns:
        if df[col].dtype in [object, 'string']:
            if df[col].astype(str).str.match(pattern).any():
                matching_columns.append(col)
                
    return matching_columns

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
