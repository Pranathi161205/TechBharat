# scripts/analyze_data.py

import pandas as pd

# The function now accepts threshold parameters
def analyze_data(df, kit_threshold=0.8, anc_threshold=0.5, high_risk_threshold=0.1):
    """
    Analyzes the transformed data to highlight key insights for policymakers
    using dynamic thresholds.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for analysis.")
        return None

    insights = []
    
    print("   - Analyzing key metrics and identifying areas of concern...")
    
    # Analysis 1: Identify districts with low kit coverage
    low_coverage_districts = df[df['kit_coverage_ratio'] < kit_threshold].sort_values(by='kit_coverage_ratio')
    if not low_coverage_districts.empty:
        insights.append(f"\n💡 **Actionable Insight: Districts with Kit Distribution Below {kit_threshold*100}% Threshold**")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(low_coverage_districts[['districtName', 'kit_coverage_ratio', 'total_kits_distributed']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")

    # Analysis 2: Identify districts with low ANC visit completion rates
    low_anc_rate_districts = df[df['anc4_to_anc1_ratio'] < anc_threshold].sort_values(by='anc4_to_anc1_ratio')
    if not low_anc_rate_districts.empty:
        insights.append(f"\n\n💡 **Actionable Insight: Districts with ANC Completion Below {anc_threshold*100}% Threshold**")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(low_anc_rate_districts[['districtName', 'anc4_to_anc1_ratio', 'total_anc1', 'total_anc4']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")

    # Analysis 3: Identify districts with high rates of high-risk pregnancies
    high_risk_districts = df[df['high_risk_ratio'] > high_risk_threshold].sort_values(by='high_risk_ratio', ascending=False)
    if not high_risk_districts.empty:
        insights.append(f"\n\n💡 **Actionable Insight: Districts with High-Risk Pregnancy Ratio Above {high_risk_threshold*100}% Threshold**")
        insights.append("----------------------------------------------------------------------------------------")
        insights.append(high_risk_districts[['districtName', 'high_risk_ratio', 'total_high_risk']].to_string(index=False))
        insights.append("----------------------------------------------------------------------------------------")
    
    print("Analysis complete.")
    return "\n".join(insights)

    # scripts/analyze_data.py

# ... (all your existing imports) ...
from scipy.stats import zscore # You'll need to install this library: pip install scipy

# ... (all your existing analyze_data function) ...

def find_anomalies(df, metric_col, z_score_threshold=3):
    """
    Finds and flags districts with anomalous values for a given metric.
    """
    if df is None or df.empty:
        return None, "DataFrame is empty."
    
    # Calculate the Z-score for the metric column
    df['z_score'] = zscore(df[metric_col].fillna(0))
    
    # Flag anomalies based on the absolute Z-score
    anomalies = df[abs(df['z_score']) > z_score_threshold]
    
    if not anomalies.empty:
        message = f"Found {len(anomalies)} anomalies for metric '{metric_col}'. Threshold: {z_score_threshold} std dev."
    else:
        message = f"No anomalies found for metric '{metric_col}'."
        
    return anomalies, message