import pandas as pd
import re

def normalize_column_name(name: str) -> str:
    """Normalize a dataframe column name into a predictable snake_case identifier.

    Rules:
      - Convert to string and lowercase
      - Replace common degree symbols (° and \u00b0) with 'deg'
      - Remove parentheses and other punctuation
      - Replace whitespace and hyphens with single underscore
      - Collapse multiple underscores
      - Strip leading/trailing underscores
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    # normalize unicode degree symbols to 'deg'
    s = s.replace('\u00b0', 'deg').replace('°', 'deg')
    # remove common mojibake artifacts
    s = s.replace('Â', '')
    # remove parentheses but keep their content
    s = re.sub(r"[\(\)]", "", s)
    # remove any characters that are not alphanumeric, spaces, hyphens or underscores
    s = re.sub(r"[^\w\s\-]", "", s)
    # replace spaces and hyphens with underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    s = s.strip('_')
    return s

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV file with a couple of helpful fallbacks for encodings."""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except UnicodeDecodeError:
        # common fallback
        df = pd.read_csv(file_path, encoding='latin1')
        print("Dataset loaded successfully (latin1 fallback).")
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None

def _find_normalized_col_in_df(df, configured_name):
    """Return the actual df column whose normalized form best matches configured_name (or None)."""
    if configured_name is None:
        return None
    target = normalize_column_name(configured_name)
    # direct normalized match
    for col in df.columns:
        if normalize_column_name(col) == target:
            return col
    # substring/token overlap fallback
    target_tokens = set(target.split('_'))
    best_col, best_score = None, 0
    for col in df.columns:
        tokens = set(normalize_column_name(col).split('_'))
        score = len(tokens & target_tokens)
        if score > best_score:
            best_score = score
            best_col = col
    if best_score > 0:
        return best_col
    return None

def clean_data(raw_df: pd.DataFrame, columns_config: dict = None) -> pd.DataFrame:
    """Cleans and standardizes the dataframe.

    Key behavior:
      - normalize column names up-front to snake_case-ish tokens
      - map configured column names to logical keys (e.g. 'date', 'district', metrics)
      - coerce types and strip strings
    """
    if raw_df is None:
        print("Error: No dataframe provided to clean_data.")
        return None

    df = raw_df.copy()

    # 1) Normalize all column names first (keep normalized names but keep originals as values)
    normalized_map = {col: normalize_column_name(col) for col in df.columns}
    df.rename(columns=normalized_map, inplace=True)

    # 2) Map configured names (human-readable) to logical keys expected downstream
    #    e.g. config provides {'district': 'District', 'min_temp': 'Min Temp'}.
    #    We find the actual normalized column in df and rename it to the logical key.
    if columns_config:
        # handle date and district keys first
        for logical_key in ('date', 'district'):
            configured_name = columns_config.get(logical_key)
            if configured_name:
                found = _find_normalized_col_in_df(df, configured_name)
                if found:
                    df.rename(columns={found: logical_key}, inplace=True)

        # handle metrics list if present, rename each to its normalized logical metric key
        metrics_list = columns_config.get('metrics', []) or []
        mapped_metrics = []
        for metric in metrics_list:
            found = _find_normalized_col_in_df(df, metric)
            if found:
                logical_metric = normalize_column_name(metric)
                # avoid clobbering 'date'/'district'
                if logical_metric in ('date', 'district'):
                    logical_metric = f"metric_{logical_metric}"
                df.rename(columns={found: logical_metric}, inplace=True)
                mapped_metrics.append(logical_metric)
        # update the provided config object (caller can use normalized metric names afterwards)
        columns_config['metrics'] = mapped_metrics

    # 3) Strip whitespace from string/object columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 4) Convert date-like column if renamed to 'date'
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            pass

    # 5) Optionally coerce metrics to numeric if config provided them; otherwise attempt sensible coercion
    metrics_cols = columns_config.get('metrics', []) if columns_config else []
    if metrics_cols:
        for col in metrics_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    else:
        # fallback: try to coerce any non-date/object columns where possible
        for col in df.columns:
            if col == 'date' or col == 'district':
                continue
            if df[col].dtype == 'object':
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().sum() > 0:
                    df[col] = coerced.fillna(0)

    # 6) Normalize district strings if that column exists
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).str.strip().str.title()

    # 7) Drop duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"   - Removed {initial_rows - len(df)} duplicate rows.")

    print("   - Data cleaning complete.")
    return df

# quick self-test
if __name__ == '__main__':
    sample = pd.DataFrame({
        'District ': ['A', 'B'],
        'Max Temp (Â°C)': [35, 30],
        'Min Temp': [25, 22],
        ' Some-Col ': [1, 2]
    })
    cfg = {'district': 'District', 'min_temp': 'Min Temp', 'max_temp': 'Max Temp (Â°C)'}
    cleaned = clean_data(sample, cfg)
    print('\nOriginal columns:', list(sample.columns))
    print('Cleaned columns:', list(cleaned.columns))
    print(cleaned.head())
