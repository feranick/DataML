#!/usr/bin/env python3
'''
**************************************************
* Split Columns wth "AA-BB" codes into two colums
* version: v2025.11.21.1
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
    sheet_name_new = "split"

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 SplitColumnCodesExcel.py <Excel File> <column>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    conf = Conf()
    
    print(" Opening Excel File:",sys.argv[1],"...\n")
    
    xls = pd.ExcelFile(sys.argv[1])
    
    if conf.sheet_name_new in xls.sheet_names:
        sheet_name = conf.sheet_name_new
        print(" Sheet name 'Split' found.\n")
    elif conf.sheet_name_original in xls.sheet_names:
        sheet_name = conf.sheet_name_original
        print(" Sheet name 'Split' NOT found, using 'as_received`.\n")
    else:
        sheet_name = xls.sheet_names[0]
        print(" Sheet name 'Split' NOT found, first sheet in file.\n")
    
    df_original = pd.read_excel(sys.argv[1], sheet_name=sheet_name, header=None)

    print(" Original table:\n", df_original)
    
    df_result, success = split_column(
        df=df_original.copy(),
        column_identifier = int(sys.argv[2]),
        )
    
    if success:
        print("\n Modified table:\n", df_result)
    
        with pd.ExcelWriter(sys.argv[1], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_result.to_excel(writer, sheet_name=conf.sheet_name_new, index=False, header=False)
    
        print(f"\n Table saved in: {sys.argv[1]}\n")
    else:
        print(f"\n Table not saved. Check that you are coverting the right column\n")
    
    
#************************************
# Spli individual column with codes
# into two separate columns
#************************************
def split_column(df: pd.DataFrame, column_identifier: int) -> pd.DataFrame:
    try:
        print(f"Splitting column by index:")
        column_name = df.columns[column_identifier]
        new_cols = df[column_name].str.split('-', expand=True)
    
        if len(new_cols.columns) == 1:
            return df, False
    
        new_cols.iat[0,1] = new_cols.iloc[0,0]
        
        print(new_cols.iloc[3,0])
        
        new_col_name = new_cols.iloc[3,0].split(' ')[0] + " type"
        new_cols.iat[3,1] = new_col_name
    
        df.insert(loc=column_identifier, column='Unnamed: 2a', value=new_cols.iloc[:,1])
        df.insert(loc=column_identifier, column='Unnamed: 2b', value=new_cols.iloc[:,0])
        df = df.drop(columns = [column_name])
    except IndexError as e:
        print(f"Error: {e}")
        return df, False
    return df, True

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
