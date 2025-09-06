# scripts/analyze_data.py

import pandas as pd
from scipy.stats import zscore

def analyze_data(df, kit_threshold=0.8, anc_threshold=0.5, high_risk_threshold=0.1):
    if df is None or df.empty:
        return None
    insights = []
    low_coverage_districts = df[df['kit_coverage_ratio'] < kit_threshold].sort_values(by='kit_coverage_ratio')
    if not low_coverage_districts.empty:
        insights.append(f"\n💡 **Actionable Insight: Districts with Kit Distribution Below {kit_threshold*100}% Threshold**")
        insights.append(low_coverage_districts[['districtName', 'kit_coverage_ratio']].to_string(index=False))
    low_anc_rate_districts = df[df['anc4_to_anc1_ratio'] < anc_threshold].sort_values(by='anc4_to_anc1_ratio')
    if not low_anc_rate_districts.empty:
        insights.append(f"\n\n💡 **Actionable Insight: Districts with ANC Completion Below {anc_threshold*100}% Threshold**")
        insights.append(low_anc_rate_districts[['districtName', 'anc4_to_anc1_ratio']].to_string(index=False))
    high_risk_districts = df[df['high_risk_ratio'] > high_risk_threshold].sort_values(by='high_risk_ratio', ascending=False)
    if not high_risk_districts.empty:
        insights.append(f"\n\n💡 **Actionable Insight: Districts with High-Risk Pregnancy Ratio Above {high_risk_threshold*100}% Threshold**")
        insights.append(high_risk_districts[['districtName', 'high_risk_ratio']].to_string(index=False))
    return "\n".join(insights)

def find_anomalies(df, metric_col, z_score_threshold=3):
    if df is None or df.empty:
        return None, "DataFrame is empty."
    df['z_score'] = zscore(df[metric_col].fillna(0))
    anomalies = df[abs(df['z_score']) > z_score_threshold]
    if not anomalies.empty:
        message = f"Found {len(anomalies)} anomalies for metric '{metric_col}'. Threshold: {z_score_threshold} std dev."
    else:
        message = f"No anomalies found for metric '{metric_col}'."
    return anomalies, message

def generate_executive_summary(df, districts_to_report):
    if df is None or df.empty:
        return "Error: DataFrame is empty for executive summary."
    summary_df = df[df['districtName'].isin(districts_to_report)].copy()
    summary_df = summary_df[[
        'districtName', 'kit_coverage_ratio', 'anc4_to_anc1_ratio', 'high_risk_ratio', 'gov_facility_utilization'
    ]].rename(columns={
        'kit_coverage_ratio': 'Kit Coverage', 'anc4_to_anc1_ratio': 'ANC4 Completion', 'high_risk_ratio': 'High-Risk Ratio', 'gov_facility_utilization': 'Govt Facility Use'
    })
    return summary_df.to_string(index=False)