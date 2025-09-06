import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


def clean_data(df, config_columns):
    if df is None:
        return None
    
    date_col = config_columns.get('date')
    district_col = config_columns.get('district')
    metrics_cols = config_columns.get('metrics', [])

    # Step 1: Correct Data Types
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    for col in metrics_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if district_col and district_col in df.columns:
        df[district_col] = df[district_col].astype(str).str.strip().str.title()

    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"   - Removed {initial_rows - len(df)} duplicate rows.")

    print("   - Data cleaning complete.")
    return df
