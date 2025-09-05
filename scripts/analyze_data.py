# scripts/analyze_data.py

import pandas as pd

def analyze_data(df):
    """
    Analyzes the transformed data to highlight key insights for policymakers.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for analysis.")
        return None

    insights = []
    
    print("   - Analyzing key metrics and identifying areas of concern...")

    # Define thresholds for analysis (these can be adjusted by the policymaker)
    KIT_COVERAGE_THRESHOLD = 0.8  # Districts with <80% kits per registered woman
    ANC4_RATE_THRESHOLD = 0.5     # Districts with <50% ANC4-to-ANC1 ratio
    HIGH_RISK_RATIO_THRESHOLD = 0.1 # Districts with >10% high-risk pregnancies
    
    # Analysis 1: Identify districts with low kit coverage
    low_coverage_districts = df[df['kit_coverage_ratio'] < KIT_COVERAGE_THRESHOLD].sort_values(by='kit_coverage_ratio')
    if not low_coverage_districts.empty:
        insights.append("\n💡 **Actionable Insight: Districts with Insufficient Kit Distribution**")
        insights.append(f"The following districts have a kit distribution coverage below the {KIT_COVERAGE_THRESHOLD*100}% threshold:")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(low_coverage_districts[['districtName', 'kit_coverage_ratio', 'total_kits_distributed']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")

    # Analysis 2: Identify districts with low ANC visit completion rates
    low_anc_rate_districts = df[df['anc4_to_anc1_ratio'] < ANC4_RATE_THRESHOLD].sort_values(by='anc4_to_anc1_ratio')
    if not low_anc_rate_districts.empty:
        insights.append("\n\n💡 **Actionable Insight: Districts with Gaps in Antenatal Care**")
        insights.append(f"The following districts have an ANC4 completion rate below the {ANC4_RATE_THRESHOLD*100}% threshold:")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(low_anc_rate_districts[['districtName', 'anc4_to_anc1_ratio', 'total_anc1', 'total_anc4']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")

    # Analysis 3: Identify districts with high rates of high-risk pregnancies
    high_risk_districts = df[df['high_risk_ratio'] > HIGH_RISK_RATIO_THRESHOLD].sort_values(by='high_risk_ratio', ascending=False)
    if not high_risk_districts.empty:
        insights.append("\n\n💡 **Actionable Insight: Districts with High-Risk Pregnancy Rates**")
        insights.append(f"The following districts have a high-risk pregnancy ratio above the {HIGH_RISK_RATIO_THRESHOLD*100}% threshold, requiring additional medical resources:")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(high_risk_districts[['districtName', 'high_risk_ratio', 'total_high_risk']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")
    
    print("Analysis complete.")
    return "\n".join(insights)

# Example usage for testing this script alone
if __name__ == '__main__':
    file_path = 'data/mch_kit_data_transformed.csv'
    try:
        transformed_df = pd.read_csv(file_path)
        insights_log = analyze_data(transformed_df)
        if insights_log:
            print("\n" + insights_log)
    except FileNotFoundError:
        print(f"Error: The transformed data file '{file_path}' was not found. Please run main.py first.")