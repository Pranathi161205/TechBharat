# scripts/analyze_data.py

import pandas as pd
from scipy.stats import zscore

def analyze_data(df, dataset_config):
    if df is None or df.empty:
        return None
    
    if 'health_data' in dataset_config:
        insights = []
        kit_threshold = dataset_config['default_thresholds']['kit_coverage_ratio']
        high_risk_threshold = dataset_config['default_thresholds']['high_risk_ratio']
        
        low_coverage_districts = df[df['kit_coverage_ratio'] < kit_threshold].sort_values(by='kit_coverage_ratio')
        if not low_coverage_districts.empty:
            insights.append(f"\n💡 **Actionable Insight: Districts with Kit Distribution Below {kit_threshold*100}% Threshold**")
            insights.append(low_coverage_districts[['districtName', 'kit_coverage_ratio']].to_string(index=False))
        
        high_risk_districts = df[df['high_risk_ratio'] > high_risk_threshold].sort_values(by='high_risk_ratio', ascending=False)
        if not high_risk_districts.empty:
            insights.append(f"\n\n💡 **Actionable Insight: Districts with High-Risk Pregnancy Ratio Above {high_risk_threshold*100}% Threshold**")
            insights.append(high_risk_districts[['districtName', 'high_risk_ratio']].to_string(index=False))

        return "\n".join(insights)
    
    elif 'temperature_data' in dataset_config:
        insights = []
        temp_threshold = dataset_config['default_thresholds']['temp_range']

        hot_districts = df[df['temp_range'] > temp_threshold]
        if not hot_districts.empty:
            insights.append(f"\n💡 **Actionable Insight: Districts with Temperature Range Above {temp_threshold}°C**")
            insights.append(hot_districts[['District', 'temp_range']].to_string(index=False))
        
        return "\n".join(insights)
    
    return "Analysis not defined for this dataset."

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
        'districtName', 'kit_coverage_ratio', 'anc4_to_anc1_ratio', 'high_risk_ratio', 'gov_facility_utilization', 'anc2_to_anc1_ratio'
    ]].rename(columns={
        'kit_coverage_ratio': 'Kit Coverage', 'anc4_to_anc1_ratio': 'ANC4 Completion', 'high_risk_ratio': 'High-Risk Ratio', 'gov_facility_utilization': 'Govt Facility Use', 'anc2_to_anc1_ratio': 'ANC2 Follow-up'
    })
    return summary_df.to_string(index=False)

def run_root_cause_analysis(df, problem_metric):
    if df is None or df.empty:
        return "Error: DataFrame is empty for root cause analysis."
    numerical_df = df.select_dtypes(include=['number'])
    correlation_matrix = numerical_df.corr()
    correlations = correlation_matrix[problem_metric].sort_values(ascending=False)
    return correlations.to_string()