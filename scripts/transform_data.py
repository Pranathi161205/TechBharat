# scripts/transform_data.py

import pandas as pd

def transform_data(df):
    """
    Aggregates cleaned data and generates key metrics and ratios.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for transformation.")
        return None

    print("   - Aggregating data by district...")
    # Aggregate counts by district
    aggregated_df = df.groupby('districtName').agg(
        total_pw_registered=('pwRegCnt', 'sum'),
        total_kits_distributed=('kitsCnt', 'sum'),
        total_deliveries=('delCnt', 'sum'),
        total_gov_deliveries=('govtDelCnt', 'sum'),
        total_pvt_deliveries=('pvtDelCnt', 'sum'),
        total_anc1=('anc1Cnt', 'sum'),
        total_anc4=('anc4Cnt', 'sum'),
        total_high_risk=('highRiskCnt', 'sum'),
        total_immunizations=('chImzCnt', 'sum')
    ).reset_index()

    print("   - Generating ratios and coverage metrics...")
    # Generate ratios and coverage metrics
    # Kit Distribution Coverage: Kits per registered pregnant woman
    aggregated_df['kit_coverage_ratio'] = (aggregated_df['total_kits_distributed'] / aggregated_df['total_pw_registered']).round(2)
    
    # Government Facility Utilization Ratio: Gov Deliveries vs Total Deliveries
    aggregated_df['gov_facility_utilization'] = (aggregated_df['total_gov_deliveries'] / aggregated_df['total_deliveries']).round(2)
    
    # ANC Visit Completion Rate: Ratio of 4th ANC visit to 1st ANC visit
    # Handle division by zero for districts with no anc1 visits
    aggregated_df['anc4_to_anc1_ratio'] = (aggregated_df['total_anc4'] / aggregated_df['total_anc1']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)
    
    # High-Risk Pregnancy Ratio
    aggregated_df['high_risk_ratio'] = (aggregated_df['total_high_risk'] / aggregated_df['total_pw_registered']).round(2)

    print("Data transformation complete.")
    return aggregated_df