#!/usr/bin/env python3
'''
**************************************************
* Create Master Dataset from provided xlsx
* This version includes: 
* Split Columns wth "AA-BB" codes into two colums
* version: v2025.11.22.1
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
    
    xls = pd.ExcelFile(sys.argv[1])
    
    if conf.sheet_name_new in xls.sheet_names:
        sheet_name = conf.sheet_name_new
        print(f" Sheet name '{conf.sheet_name_new}' found.\n")
    elif conf.sheet_name_original in xls.sheet_names:
        sheet_name = conf.sheet_name_original
        print(f" Sheet name '{conf.sheet_name_new}' NOT found, using '{conf.sheet_name_original}'.\n")
    else:
        sheet_name = xls.sheet_names[0]
        print(f" Sheet name '{conf.sheet_name_new}' NOT found, first sheet in file.\n")
    
    df_original = pd.read_excel(sys.argv[1], sheet_name=sheet_name, header=None)

    print(" Original table:\n", df_original)
    
    if len(find_formatted_columns(df_original)) == 0:
        print(f"\n No columns with 'AA-BB' codes found.\n")
        return
        
    print(f"\n Columns with 'AA-BB' codes: {find_formatted_columns(df_original)}")
    offset = 0
    df_temp = df_original.copy()
    
    for col in find_formatted_columns(df_original):
        print(f"\n Processing column {col}")
        df_temp, success = split_column(
            df=df_temp,
            #column_identifier = int(sys.argv[2]),
            column_identifier = col + offset
            )
        offset += 1
        df_result = df_temp.copy()
    
    if success:
        if conf.skip_NaN_rows:
            df_result = df_result.dropna(how='all', axis=0)
    
        if conf.replace_NaN:
            df_result.iloc[conf.rows_to_skip:,conf.cols_to_skip:] = df_result.iloc[conf.rows_to_skip:, conf.cols_to_skip:].fillna(conf.value_for_NaN)
            
        if conf.drop_first_column:
            print("\n Dropping first column")
            df_result = df_result.drop(df_result.columns[0], axis=1)
            
        print("\n Modified table:\n", df_result)
    
        with pd.ExcelWriter(sys.argv[1], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_result.to_excel(writer, sheet_name=conf.sheet_name_new, index=False, header=False)
    
        print(f"\n Table saved in: {sys.argv[1]}\n")
    else:
        print(f"\n Table not saved. Check that you are coverting the right column\n")
    
    
#************************************
# Split individual column with codes
# into two separate columns
#************************************
def split_column(df: pd.DataFrame, column_identifier: int) -> pd.DataFrame:
    conf = Conf()
    try:
        column_name = df.columns[column_identifier]
        # Create new split columns
        new_cols = df[column_name].str.split('-', expand=True)
    
        # If the target column does not have AA-BB codes, exit
        if len(new_cols.columns) == 1:
            return df, False
    
        # Provide correct naming to new columns
        new_col_name = new_cols.iloc[conf.row_with_name,0].rsplit(' ',1)[0] + " type"
        new_cols.iat[conf.row_with_name,1] = new_col_name
    
        df.insert(loc=column_identifier, column=str(column_identifier)+'a', value=new_cols.iloc[:,1])
        df.insert(loc=column_identifier, column=str(column_identifier)+'b', value=new_cols.iloc[:,0])
        df = df.drop(columns = [column_name])
    except IndexError as e:
        print(f"Error: {e}")
        return df, False
    return df, True

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
