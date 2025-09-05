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
    numerical_cols = ['pwregcnt', 'pwtrkcnt', 'delcnt', 'anc1cnt', 'anc2cnt', 'anc3cnt', 'anc4cnt', 'chimzcnt', 'highriskcnt', 'govtdelcnt', 'pvtdelcnt', 'kitscnt']
    for col in numerical_cols:
        if col in df.columns:
            # Ensure the column is numeric before filling NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Standardize Text Data: Convert district and mandal names to consistent format (Title Case)
    for col in ['districtname', 'mandalname', 'villagename']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            
    # Correct Data Types: Convert datadate to a proper datetime object
    if 'datadate' in df.columns:
        df['datadate'] = pd.to_datetime(df['datadate'], errors='coerce')

    # Remove Duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"   - Removed {rows_removed} duplicate rows.")

    print("   - Data cleaning complete.")
    return df