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
    aggregated_df = df.groupby('districtname').agg(
        total_pw_registered=('pwregcnt', 'sum'),
        total_kits_distributed=('kitscnt', 'sum'),
        total_deliveries=('delcnt', 'sum'),
        total_gov_deliveries=('govtdelcnt', 'sum'),
        total_pvt_deliveries=('pvtdelcnt', 'sum'),
        total_anc1=('anc1cnt', 'sum'),
        total_anc4=('anc4cnt', 'sum'),
        total_high_risk=('highriskcnt', 'sum'),
        total_immunizations=('chimzcnt', 'sum')
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

# Example usage (for testing this script alone)
if __name__ == '__main__':
    # Assuming the cleaned data exists
    file_path = 'data/mch_kit_data_cleaned.csv'
    try:
        cleaned_df = pd.read_csv(file_path)
        transformed_df = transform_data(cleaned_df)
        if transformed_df is not None:
            print("\nTransformed DataFrame (head):")
            print(transformed_df.head())
            
    except FileNotFoundError:
        print(f"Error: The cleaned data file '{file_path}' was not found. Please run main.py first.")