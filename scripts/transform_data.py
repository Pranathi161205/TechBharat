# scripts/transform_data.py

import pandas as pd

def transform_data(df, dataset_config, dataset_name):
    """
    Transforms the dataframe based on dataset type.

    Args:
        df (pd.DataFrame): Cleaned dataframe
        dataset_config (dict): Config for the dataset (from config.yaml)
        dataset_name (str): Name of the dataset

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for transformation.")
        return None

    # -----------------------
    # Health Data Transformation
    # -----------------------
    if dataset_name == "health_data":
        district_col = dataset_config['columns'].get('district')
        if district_col is None:
            print("Error: 'district' column not found in config for health data.")
            return None

        aggregated_df = df.groupby(district_col).agg(
            total_pw_registered=('pwRegCnt', 'sum'),
            total_kits_distributed=('kitsCnt', 'sum'),
            total_deliveries=('delCnt', 'sum'),
            total_gov_deliveries=('govtDelCnt', 'sum'),
            total_pvt_deliveries=('pvtDelCnt', 'sum'),
            total_anc1=('anc1Cnt', 'sum'),
            total_anc2=('anc2Cnt', 'sum'),
            total_anc4=('anc4Cnt', 'sum'),
            total_high_risk=('highRiskCnt', 'sum'),
            total_immunizations=('chImzCnt', 'sum')
        ).reset_index()

        # Derived metrics
        aggregated_df['kit_coverage_ratio'] = (
            aggregated_df['total_kits_distributed'] / aggregated_df['total_pw_registered']
        ).round(2)

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
    # Temperature Data Transformation
    # -----------------------
    elif dataset_name == "temperature_data":
        district_col = dataset_config['columns'].get('district')
        temp_col = dataset_config['columns'].get('temperature')

        if district_col is None or temp_col is None:
            print("Error: 'district' or 'temperature' column not found in config for temperature dataset.")
            return None

        aggregated_df = df.groupby(district_col).agg(
            avg_temperature=(temp_col, 'mean'),
            max_temperature=(temp_col, 'max'),
            min_temperature=(temp_col, 'min')
        ).reset_index()

        aggregated_df['temp_range'] = aggregated_df['max_temperature'] - aggregated_df['min_temperature']

        print("✅ Data transformation complete for temperature data.")
        return aggregated_df

    # -----------------------
    # Unknown Dataset
    # -----------------------
    else:
        print(f"⚠️ No recognized dataset type found for '{dataset_name}'.")
        return None


# -----------------------
# Standalone Test
# -----------------------
if __name__ == "__main__":
    # Health dummy data
    health_df = pd.DataFrame({
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

    # Temperature dummy data
    temp_df = pd.DataFrame({
        'District': ['X', 'X', 'Y'],
        'Temperature': [35, 30, 28]
    })

    config_health = {
        'columns': {'district': 'districtName'}
    }
    config_temp = {
        'columns': {'district': 'District', 'temperature': 'Temperature'}
    }

    print("\n--- Health Data Transformation ---")
    print(transform_data(health_df, config_health, "health_data"))

    print("\n--- Temperature Data Transformation ---")
    print(transform_data(temp_df, config_temp, "temperature_data"))
