# scripts/clean_data.py

import pandas as pd

def load_data(file_path):
    """Loads the dataset from a specified file path."""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def clean_data(df):
    """Performs data cleaning and standardization."""
    if df is None:
        return None

    # Handle Missing Values: Fill numerical NaNs with 0
    numerical_cols = ['pwRegCnt', 'pwTrkCnt', 'delCnt', 'anc1Cnt', 'anc2Cnt', 'anc3Cnt', 'anc4Cnt', 'chimzCnt', 'highriskCnt', 'govtDelCnt', 'pvtDelCnt', 'kitsCnt']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Standardize Text Data: Convert district and mandal names to consistent format (Title Case)
    for col in ['districtName', 'mandalName', 'villageName']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            
    # Correct Data Types: Convert dataDate to a proper datetime object
    if 'dataDate' in df.columns:
        df['dataDate'] = pd.to_datetime(df['dataDate'], errors='coerce')

    # Remove Duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"   - Removed {rows_removed} duplicate rows.")

    print("   - Data cleaning complete.")
    return df