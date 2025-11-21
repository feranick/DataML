#!/usr/bin/env python3
'''
**************************************************
* Split Columns wth "AA-BB" codes into two colums
* version: v2025.11.20.1
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************
'''
print(__doc__)
import pandas as pd
import sys, os.path, h5py

#************************************
# Parameters definition
#************************************
class dP:
    saveAsCsv = True
    saveAsHDF = False

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 SplitColumnCodesExcel.py <Excel File> <column>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    
    print(" Opening Excel File:",sys.argv[1],"...\n")
    
    xls = pd.ExcelFile(sys.argv[1])
    
    if "split" in xls.sheet_names:
        sheet_name = "split"
        print(" Sheet name 'Split' found.\n")
    else:
        sheet_name = "as_received"
        print(" Sheet name 'Split' NOT found, using 'as_received`.\n")
    
    df_original = pd.read_excel(sys.argv[1], sheet_name=sheet_name, header=None)

    print(" Original table:\n", df_original)
    
    df_result = split_date_part(
        df=df_original.copy(),
        column_identifier = int(sys.argv[2]),
        )
    
    print("\n Modified table:\n", df_result)
    
    with pd.ExcelWriter(sys.argv[1], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        df_result.to_excel(writer, sheet_name="split", index=False, header=False)
    
    print(f"\n Table saved in: {sys.argv[1]}\n")
    
def split_date_part(df: pd.DataFrame, column_identifier: int) -> pd.DataFrame:
    try:
    
        print(f"Splitting column by index:")
        column_name = df.columns[column_identifier]
        new_cols = df[column_name].str.split('-', expand=True)
    
        new_cols.iat[0,1] = new_cols.iloc[0,0]
        
        print(new_cols.iloc[3,0])
        
        new_col_name = new_cols.iloc[3,0].split(' ')[0] + " type"
        new_cols.iat[3,1] = new_col_name
    
        df.insert(loc=column_identifier, column='Unnamed: 2a', value=new_cols.iloc[:,1])
        df.insert(loc=column_identifier, column='Unnamed: 2b', value=new_cols.iloc[:,0])
        df = df.drop(columns = [column_name])
    except IndexError:
        print(f"Error")
    return df

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
