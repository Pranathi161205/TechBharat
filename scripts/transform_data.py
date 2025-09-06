# scripts/transform_data.py
import pandas as pd
import re
from typing import Optional, List, Dict, Any

# -----------------------
# Utilities
# -----------------------
def normalize_column_name(name: Optional[str]) -> str:
    """Normalize a column name to snake_case and remove odd unicode artifacts."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    # replace degree symbols with 'deg'
    s = s.replace('\u00b0', 'deg').replace('°', 'deg').replace('º', 'deg')
    # remove common artefacts
    s = s.replace('Â', '')
    # remove parentheses but keep content
    s = re.sub(r"[\(\)]", "", s)
    # remove punctuation except underscore and whitespace and hyphen
    s = re.sub(r"[^\w\s\-]", "", s)
    # convert spaces and hyphens to underscore
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip('_')

def _normalize_token(s: str) -> str:
    """Simpler normalizer used for fuzzy matching."""
    return normalize_column_name(s)

def find_best_column(df: pd.DataFrame, target_names: List[str]) -> Optional[str]:
    """
    Given a dataframe and a list of potential names (in priority order), return the best matching column name
    present in df (actual column string). Matching strategy:
      1. exact match (case-sensitive)
      2. normalized exact match
      3. substring/contain match on normalized strings
      4. token overlap score
    """
    cols = list(df.columns)
    if not cols:
        return None

    # 1. exact
    for t in target_names:
        if t in cols:
            return t

    # normalized map
    norm_map = {c: _normalize_token(c) for c in cols}
    target_norms = [_normalize_token(t) for t in target_names]

    # 2. normalized exact
    for tnorm in target_norms:
        for c, cnorm in norm_map.items():
            if cnorm == tnorm:
                return c

    # 3. substring match
    for tnorm in target_norms:
        for c, cnorm in norm_map.items():
            if tnorm in cnorm or cnorm in tnorm:
                return c

    # 4. token overlap
    best, best_score = None, 0
    t_tokens = set()
    for tn in target_norms:
        t_tokens |= set(tn.split('_'))
    for c, cnorm in norm_map.items():
        c_tokens = set(cnorm.split('_'))
        score = len(c_tokens & t_tokens)
        if score > best_score:
            best_score = score
            best = c
    if best_score > 0:
        return best

    return None

def resolve_column(df: pd.DataFrame, dataset_config: Dict[str, Any], logical_key: str, candidates: List[str]=None) -> Optional[str]:
    """
    Resolve a logical column name (e.g. 'district', 'min_temp') using dataset_config mapping
    and fuzzy matching. Returns actual df column name or None.
    """
    # 1) if dataset_config provided and has mapping, use that
    if dataset_config and isinstance(dataset_config.get('columns'), dict):
        conf_name = dataset_config['columns'].get(logical_key)
        if conf_name:
            # if config value exactly present in df columns return it
            if conf_name in df.columns:
                return conf_name
            # try normalized match for configured name
            cfg_norm = _normalize_token(conf_name)
            for c in df.columns:
                if _normalize_token(c) == cfg_norm:
                    return c
    # 2) candidates list (common names) - try to find best
    if candidates:
        found = find_best_column(df, candidates)
        if found:
            return found
    # 3) fall back to finding column named same as logical_key
    found = find_best_column(df, [logical_key, logical_key.replace('_',' '), logical_key.replace('_','')])
    return found

# -----------------------
# Transform function
# -----------------------
def transform_data(df: pd.DataFrame, dataset_config: Dict[str, Any], dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Transforms the dataframe based on dataset type.
    Args:
        df (pd.DataFrame): cleaned dataframe (columns may be normalized already)
        dataset_config (dict): dataset config from config.yaml (may be None)
        dataset_name (str): name of the dataset (controls transform logic)
    Returns:
        pd.DataFrame or None
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for transformation.")
        return None

    # -----------------------
    # Health Data Transformation
    # -----------------------
    if dataset_name == "health_data":
        district_col = resolve_column(df, dataset_config, 'district', ['district', 'district_name', 'districtname'])
        if district_col is None:
            print("Error: 'district' column not found in config for health data.")
            return None

        # expected metric names - try common variations
        metrics_map = {
            'pwRegCnt': resolve_column(df, dataset_config, 'pwRegCnt', ['pw_reg_cnt', 'pwregcnt', 'pw_registrations', 'pw_registered']),
            'kitsCnt': resolve_column(df, dataset_config, 'kitsCnt', ['kitscnt', 'kits', 'kits_distributed', 'kits_cnt']),
            'delCnt': resolve_column(df, dataset_config, 'delCnt', ['delcnt', 'deliveries', 'delivery_count']),
            'govtDelCnt': resolve_column(df, dataset_config, 'govtDelCnt', ['govtdelcnt', 'govt_deliveries']),
            'pvtDelCnt': resolve_column(df, dataset_config, 'pvtDelCnt', ['pvtdelcnt', 'private_deliveries']),
            'anc1Cnt': resolve_column(df, dataset_config, 'anc1Cnt', ['anc1cnt', 'anc1']),
            'anc2Cnt': resolve_column(df, dataset_config, 'anc2Cnt', ['anc2cnt', 'anc2']),
            'anc4Cnt': resolve_column(df, dataset_config, 'anc4Cnt', ['anc4cnt', 'anc4']),
            'highRiskCnt': resolve_column(df, dataset_config, 'highRiskCnt', ['highriskcnt', 'high_risk']),
            'chImzCnt': resolve_column(df, dataset_config, 'chImzCnt', ['chimzcnt', 'child_immunizations', 'immunizations'])
        }

        # ensure numeric coercions for found metrics
        for k, col in metrics_map.items():
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        aggregated_df = df.groupby(district_col).agg(
            total_pw_registered=(metrics_map.get('pwRegCnt'), 'sum') if metrics_map.get('pwRegCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_kits_distributed=(metrics_map.get('kitsCnt'), 'sum') if metrics_map.get('kitsCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_deliveries=(metrics_map.get('delCnt'), 'sum') if metrics_map.get('delCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_gov_deliveries=(metrics_map.get('govtDelCnt'), 'sum') if metrics_map.get('govtDelCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_pvt_deliveries=(metrics_map.get('pvtDelCnt'), 'sum') if metrics_map.get('pvtDelCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_anc1=(metrics_map.get('anc1Cnt'), 'sum') if metrics_map.get('anc1Cnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_anc2=(metrics_map.get('anc2Cnt'), 'sum') if metrics_map.get('anc2Cnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_anc4=(metrics_map.get('anc4Cnt'), 'sum') if metrics_map.get('anc4Cnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_high_risk=(metrics_map.get('highRiskCnt'), 'sum') if metrics_map.get('highRiskCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size'),
            total_immunizations=(metrics_map.get('chImzCnt'), 'sum') if metrics_map.get('chImzCnt') else pd.NamedAgg(column=df.columns[0], aggfunc='size')
        ).reset_index()

        # Derived metrics (protect against division by zero)
        aggregated_df['kit_coverage_ratio'] = (aggregated_df['total_kits_distributed'] / aggregated_df['total_pw_registered']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        aggregated_df['gov_facility_utilization'] = (aggregated_df['total_gov_deliveries'] / aggregated_df['total_deliveries']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        aggregated_df['anc4_to_anc1_ratio'] = (aggregated_df['total_anc4'] / aggregated_df['total_anc1']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        aggregated_df['high_risk_ratio'] = (aggregated_df['total_high_risk'] / aggregated_df['total_pw_registered']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        aggregated_df['anc2_to_anc1_ratio'] = (aggregated_df['total_anc2'] / aggregated_df['total_anc1']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        print("✅ Data transformation complete for health data.")
        return aggregated_df

    # -----------------------
    # Temperature Data Transformation
    # -----------------------
    elif dataset_name == "temperature_data":
        district_col = resolve_column(df, dataset_config, 'district', ['district', 'district_name'])
        min_col = resolve_column(df, dataset_config, 'min_temp', ['min_temp', 'min temp', 'minimum_temperature', 'min_temperature'])
        max_col = resolve_column(df, dataset_config, 'max_temp', ['max_temp', 'max temp', 'maximum_temperature', 'max_temperature'])

        if district_col is None or min_col is None or max_col is None:
            print("Error: 'district', 'min_temp', or 'max_temp' column not found in config for temperature dataset.")
            return None

        # coerce
        df[min_col] = pd.to_numeric(df[min_col], errors='coerce')
        df[max_col] = pd.to_numeric(df[max_col], errors='coerce')

        aggregated_df = df.groupby(district_col).agg(
            min_temperature=(min_col, 'min'),
            max_temperature=(max_col, 'max')
        ).reset_index()

        # ensure columns names and compute derived
        aggregated_df['avg_temp'] = ((aggregated_df['min_temperature'] + aggregated_df['max_temperature']) / 2).round(2)
        aggregated_df['temp_range'] = (aggregated_df['max_temperature'] - aggregated_df['min_temperature']).round(2)

        print("✅ Data transformation complete for temperature data.")
        return aggregated_df

    # -----------------------
    # Anganwadi Data Transformation
    # -----------------------
    elif dataset_name == "anganwadi_data":
        district_col = resolve_column(df, dataset_config, 'district', ['district', 'district_name'])
        awc_id_col = resolve_column(df, dataset_config, 'awc_id', ['awc_id', 'awc id', 'awc_id_no', 'center_id', 'awc'])
        child_col = resolve_column(df, dataset_config, 'child_enrolled', ['children_enrolled', 'child_enrolled', 'children'])
        plw_col = resolve_column(df, dataset_config, 'plw_enrolled', ['plw_enrolled', 'plw', 'pregnant_lactating'])

        if district_col is None:
            print("Error: 'district' column not found in config for anganwadi data.")
            return None

        # coerce numeric if present
        if child_col and child_col in df.columns:
            df[child_col] = pd.to_numeric(df[child_col], errors='coerce').fillna(0)
        if plw_col and plw_col in df.columns:
            df[plw_col] = pd.to_numeric(df[plw_col], errors='coerce').fillna(0)

        # total centers
        if awc_id_col and awc_id_col in df.columns:
            centers = df.groupby(district_col)[awc_id_col].nunique().reset_index(name='total_centers')
        else:
            centers = df.groupby(district_col).size().reset_index(name='total_centers')

        parts = [centers.set_index(district_col)]
        if child_col and child_col in df.columns:
            parts.append(df.groupby(district_col)[child_col].sum().rename('total_children_enrolled'))
        if plw_col and plw_col in df.columns:
            parts.append(df.groupby(district_col)[plw_col].sum().rename('total_plw_enrolled'))

        aggregated_df = pd.concat(parts, axis=1).reset_index()

        if 'total_children_enrolled' in aggregated_df.columns and 'total_centers' in aggregated_df.columns:
            aggregated_df['avg_children_per_center'] = (aggregated_df['total_children_enrolled'] / aggregated_df['total_centers']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        print("✅ Data transformation complete for anganwadi data.")
        return aggregated_df

    # -----------------------
    # Skill Development / Training Data Transformation
    # -----------------------
    elif dataset_name == "skill_development":
        district_col = resolve_column(df, dataset_config, 'district', ['district', 'district_name'])
        participants_col = resolve_column(df, dataset_config, 'participants', ['participants', 'participant_count', 'total_participants', 'no_of_participants'])
        male_col = resolve_column(df, dataset_config, 'male', ['male', 'male_participants'])
        female_col = resolve_column(df, dataset_config, 'female', ['female', 'female_participants'])
        program_col = resolve_column(df, dataset_config, 'program', ['program', 'scheme', 'course', 'training_program'])
        date_col = resolve_column(df, dataset_config, 'date', ['date', 'training_date', 'month', 'start_date'])

        if district_col is None:
            print("Error: 'district' column not found in config/data for skill_development.")
            return None

        for c in (participants_col, male_col, female_col):
            if c and c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        agg_dict = {}
        if participants_col:
            agg_dict['total_participants'] = (participants_col, 'sum')
        if male_col:
            agg_dict['total_male_participants'] = (male_col, 'sum')
        if female_col:
            agg_dict['total_female_participants'] = (female_col, 'sum')

        # program_count = number of training entries per district
        agg_df = df.groupby(district_col).agg(
            total_trainings=(program_col if program_col and program_col in df.columns else district_col, 'count'),
            **agg_dict
        ).reset_index()

        # normalize column names
        if district_col in agg_df.columns:
            agg_df.rename(columns={district_col: 'district'}, inplace=True)
        else:
            agg_df = agg_df.rename_axis('district').reset_index()

        agg_df['districtName'] = agg_df['district'].astype(str).str.title()

        # derived metrics
        if 'total_participants' in agg_df.columns and 'total_trainings' in agg_df.columns:
            agg_df['avg_participants_per_training'] = (agg_df['total_participants'] / agg_df['total_trainings']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        if 'total_participants' in agg_df.columns and 'total_male_participants' in agg_df.columns:
            agg_df['male_ratio'] = (agg_df['total_male_participants'] / agg_df['total_participants']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
        if 'total_participants' in agg_df.columns and 'total_female_participants' in agg_df.columns:
            agg_df['female_ratio'] = (agg_df['total_female_participants'] / agg_df['total_participants']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        print("✅ Data transformation complete for skill_development data.")
        return agg_df

    # -----------------------
    # Tourism Domestic Visitors Transformation
    # -----------------------
    elif dataset_name == "tourism_domestic":
        district_col = resolve_column(df, dataset_config, 'district', ['district', 'district_name'])
        month_col = resolve_column(df, dataset_config, 'date', ['month', 'month_year', 'date', 'month_name'])
        visitors_col = resolve_column(df, dataset_config, 'visitors', ['visitors', 'visitor_count', 'num_visitors', 'total_visitors'])

        if district_col is None or month_col is None or visitors_col is None:
            print("Error: required column(s) not found for tourism_domestic (need district, month, visitors).")
            return None

        # coerce numeric visitors
        df[visitors_col] = pd.to_numeric(df[visitors_col], errors='coerce').fillna(0)

        # aggregate metrics
        agg = df.groupby(district_col).agg(
            total_visitors=(visitors_col, 'sum'),
            avg_monthly_visitors=(visitors_col, 'mean'),
            max_monthly_visitors=(visitors_col, 'max'),
            observation_count=(visitors_col, 'count')
        ).reset_index()

        # normalize district name column to 'district' and add 'districtName'
        agg = agg.rename(columns={district_col: 'district'}) if district_col in agg.columns else agg.rename_axis('district').reset_index()
        agg['districtName'] = agg['district'].astype(str).str.title()

        # find peak month per district
        try:
            idx = df.groupby(district_col)[visitors_col].idxmax()
            peak = df.loc[idx, [district_col, month_col, visitors_col]].rename(columns={month_col: 'peak_month', visitors_col: 'peak_month_visitors'})
            peak = peak.rename(columns={district_col: 'district'})
            peak['district'] = peak['district'].astype(str)
            merged = agg.merge(peak[['district', 'peak_month', 'peak_month_visitors']], on='district', how='left')
        except Exception:
            merged = agg.copy()
            merged['peak_month'] = None
            merged['peak_month_visitors'] = 0

        merged['avg_monthly_visitors'] = merged['avg_monthly_visitors'].round(2)
        merged['total_visitors'] = merged['total_visitors'].astype(int)
        merged['peak_month_visitors'] = merged.get('peak_month_visitors', 0).fillna(0).astype(int)

        print("✅ Data transformation complete for tourism_domestic.")
        return merged

    # -----------------------
    # Unknown Dataset
    # -----------------------
    else:
        print(f"⚠️ No recognized dataset type found for '{dataset_name}'.")
        return None


# If run directly, perform a tiny self-check (do NOT assume any local CSVs)
if __name__ == "__main__":
    print("transform_data.py loaded as script. This module provides transform_data(df, dataset_config, dataset_name).")
