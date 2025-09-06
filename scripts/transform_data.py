import pandas as pd
import re

def _normalize_name(name: str) -> str:
    """Normalize a name for matching (similar to clean_data.normalize_column_name)."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = s.replace('\u00b0', 'deg').replace('°', 'deg')
    s = s.replace('Â', '')
    s = re.sub(r"[\(\)]", "", s)
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip('_')

def _find_col(df_cols, variants):
    """Find best matching actual column name from df_cols given a list of candidate variants."""
    normalized_map = { _normalize_name(c): c for c in df_cols }
    for v in variants:
        nv = _normalize_name(v)
        if nv in normalized_map:
            return normalized_map[nv]
    # substring/token overlap
    for v in variants:
        nv = _normalize_name(v)
        for nc, orig in normalized_map.items():
            if nv in nc or nc in nv:
                return orig
    # token overlap scoring
    target_tokens = set().union(*[_normalize_name(v).split('_') for v in variants])
    best = None
    best_score = 0
    for nc, orig in normalized_map.items():
        tokens = set(nc.split('_'))
        score = len(tokens & target_tokens)
        if score > best_score:
            best_score = score
            best = orig
    if best_score > 0:
        return best
    return None

def transform_data(df, dataset_config, dataset_name):
    """
    Transforms data for supported dataset types:
      - health_data: aggregates counts by district and computes ratios.
      - temperature_data: computes min/max/avg/temp_range by district.
    The function is robust to column name variations by using normalized matching.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for transformation.")
        return None

    df_cols = list(df.columns)

    def get_col(key_variants):
        """Try dataset_config mapping first, then candidate variants."""
        cols_map = dataset_config.get('columns', {}) if dataset_config else {}
        # prepare candidate list
        candidates = []
        # if key_variants is a string, allow list
        if isinstance(key_variants, str):
            key_variants = [key_variants]
        for kv in key_variants:
            candidates.append(kv)
            # if dataset_config maps logical key to a raw header, include that
            if kv in cols_map:
                candidates.append(cols_map[kv])
        # also include the normalized logical key itself as a candidate
        candidates.extend(key_variants)
        return _find_col(df_cols, candidates)

    # -----------------------
    # Health Data
    # -----------------------
    if dataset_name == "health_data":
        district_col = get_col(['district', 'districtname', 'district_name'])
        if district_col is None:
            print("Error: 'district' column not found in config/data for health_data.")
            return None

        # expected metrics with possible variants
        expected = {
            'pwRegCnt': ['pwRegCnt', 'pw_reg_cnt', 'pwregcnt', 'pw_reg', 'pw_registered'],
            'kitsCnt': ['kitsCnt', 'kits_cnt', 'kitscnt', 'kits_distributed'],
            'delCnt': ['delCnt', 'del_cnt', 'delcnt', 'deliveries', 'total_deliveries'],
            'govtDelCnt': ['govtDelCnt', 'govt_del_cnt', 'govtdelcnt', 'govt_deliveries'],
            'pvtDelCnt': ['pvtDelCnt', 'pvt_del_cnt', 'pvtdelcnt', 'private_deliveries'],
            'anc1Cnt': ['anc1Cnt', 'anc1_cnt', 'anc_1_cnt', 'anc1'],
            'anc2Cnt': ['anc2Cnt', 'anc2_cnt', 'anc_2_cnt', 'anc2'],
            'anc4Cnt': ['anc4Cnt', 'anc4_cnt', 'anc_4_cnt', 'anc4'],
            'highRiskCnt': ['highRiskCnt', 'high_risk_cnt', 'highriskcnt', 'high_risk'],
            'chImzCnt': ['chImzCnt', 'ch_imz_cnt', 'chimzcnt', 'child_immunizations']
        }

        actual = {}
        for logical, variants in expected.items():
            col = get_col(variants)
            if col is None:
                print(f"Error: Expected column for '{logical}' not found. Tried: {variants}")
                return None
            actual[logical] = col

        aggregated_df = df.groupby(district_col).agg(
            total_pw_registered=(actual['pwRegCnt'], 'sum'),
            total_kits_distributed=(actual['kitsCnt'], 'sum'),
            total_deliveries=(actual['delCnt'], 'sum'),
            total_gov_deliveries=(actual['govtDelCnt'], 'sum'),
            total_pvt_deliveries=(actual['pvtDelCnt'], 'sum'),
            total_anc1=(actual['anc1Cnt'], 'sum'),
            total_anc2=(actual['anc2Cnt'], 'sum'),
            total_anc4=(actual['anc4Cnt'], 'sum'),
            total_high_risk=(actual['highRiskCnt'], 'sum'),
            total_immunizations=(actual['chImzCnt'], 'sum')
        ).reset_index()

        # safe derived metrics
        aggregated_df['kit_coverage_ratio'] = (
            aggregated_df['total_kits_distributed'] / aggregated_df['total_pw_registered']
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        aggregated_df['gov_facility_utilization'] = (
            aggregated_df['total_gov_deliveries'] / aggregated_df['total_deliveries']
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        aggregated_df['anc4_to_anc1_ratio'] = (
            aggregated_df['total_anc4'] / aggregated_df['total_anc1']
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        aggregated_df['high_risk_ratio'] = (
            aggregated_df['total_high_risk'] / aggregated_df['total_pw_registered']
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        aggregated_df['anc2_to_anc1_ratio'] = (
            aggregated_df['total_anc2'] / aggregated_df['total_anc1']
        ).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        print("✅ Data transformation complete for health data.")
        return aggregated_df

    # -----------------------
    # Temperature Data
    # -----------------------
    elif dataset_name == "temperature_data":
        district_col = get_col(['district', 'mandal', 'districtname'])
        min_temp_col = get_col(['min_temp', 'min temp', 'mintemp', 'min_temperature', 'min_temperature_c'])
        max_temp_col = get_col(['max_temp', 'max temp', 'maxtemp', 'max_temperature', 'max_temperature_c'])

        if district_col is None or min_temp_col is None or max_temp_col is None:
            print("Error: 'district', 'min_temp', or 'max_temp' column not found in config/data for temperature_data.")
            print(f"Available columns: {df_cols}")
            return None

        aggregated_df = df.groupby(district_col).agg(
            min_temperature=(min_temp_col, 'min'),
            max_temperature=(max_temp_col, 'max')
        ).reset_index()

        aggregated_df['avg_temp'] = ((aggregated_df['min_temperature'] + aggregated_df['max_temperature']) / 2).round(2)
        aggregated_df['temp_range'] = (aggregated_df['max_temperature'] - aggregated_df['min_temperature']).round(2)

        print("✅ Data transformation complete for temperature data.")
        return aggregated_df

    else:
        print(f"⚠️ No recognized dataset type found for '{dataset_name}'.")
        return None

# quick self-test
if __name__ == '__main__':
    # health sample
    h = pd.DataFrame({
        'districtName': ['A', 'A', 'B'],
        'pwRegCnt': [10, 20, 15],
        'kitsCnt': [8, 15, 10],
        'delCnt': [5, 12, 9],
        'govtDelCnt': [3, 7, 4],
        'pvtDelCnt': [2, 5, 5],
        'anc1Cnt': [10, 18, 12],
        'anc2Cnt': [8, 15, 10],
        'anc4Cnt': [6, 10, 8],
        'highRiskCnt': [1, 2, 1],
        'chImzCnt': [5, 12, 7]
    })
    cfg_h = {'columns': {'district': 'districtName'}}
    print(transform_data(h, cfg_h, 'health_data'))

    # temperature sample
    t = pd.DataFrame({
        'District': ['X', 'X', 'Y'],
        'Min Temp': [28, 25, 22],
        'Max Temp (Â°C)': [35, 30, 28]
    })
    cfg_t = {'columns': {'district': 'District', 'min_temp': 'Min Temp', 'max_temp': 'Max Temp (Â°C)'}}
    print(transform_data(t, cfg_t, 'temperature_data'))
