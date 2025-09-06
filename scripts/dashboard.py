# scripts/dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def create_dashboard(df, metrics_to_plot, output_path='data/dashboard.png'):
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for dashboard creation.")
        return
    try:
        num_metrics = len(metrics_to_plot)
        if num_metrics == 0:
            print("No metrics specified for dashboard.")
            return
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(12, 5 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        fig.suptitle('RTGS Agent: Key Health Metrics Dashboard', fontsize=16)
        for i, metric in enumerate(metrics_to_plot):
            if metric not in df.columns:
                print(f"Warning: Metric '{metric}' not found. Skipping plot.")
                continue
            title_map = {
                'kit_coverage_ratio': 'MCH Kit Coverage Ratio',
                'high_risk_ratio': 'High-Risk Pregnancy Ratio',
                'anc4_to_anc1_ratio': 'ANC4 Completion Rate',
                'gov_facility_utilization': 'Government Facility Utilization',
                'anc2_to_anc1_ratio': 'ANC2 Follow-up Rate'
            }
            df_sorted = df.sort_values(metric, ascending=False).tail(10)
            axes[i].barh(df_sorted['districtName'], df_sorted[metric], color='skyblue')
            axes[i].set_title(f'Top 10 Districts by {title_map.get(metric, metric)}')
            axes[i].set_xlabel(title_map.get(metric, metric))
            for index, value in enumerate(df_sorted[metric]):
                axes[i].text(value, index, f' {value}', va='center')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        print(f"   - Dashboard image saved to {output_path}")
    except Exception as e:
        print(f"Error generating dashboard: {e}")

def create_district_dashboard(df, district_name, metrics_to_plot, output_path):
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided.")
        return
    try:
        num_metrics = len(metrics_to_plot)
        if num_metrics == 0:
            print("No metrics specified for the district dashboard.")
            return
        district_data = df[df['districtName'] == district_name]
        if district_data.empty:
            print(f"No data found for district '{district_name}'.")
            return
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        fig.suptitle(f'RTGS Agent: Dashboard for {district_name}', fontsize=16)
        for i, metric in enumerate(metrics_to_plot):
            if metric not in district_data.columns:
                print(f"Warning: Metric '{metric}' not found. Skipping plot.")
                continue
            value = district_data[metric].iloc[0]
            axes[i].barh([metric], [value], color='lightcoral')
            axes[i].set_title(metric)
            axes[i].set_xlabel('Value')
            axes[i].set_xlim(0, max(1.0, value * 1.2))
            axes[i].text(value, 0, f' {value}', va='center')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        print(f"   - Dashboard image saved to {output_path}")
    except Exception as e:
        print(f"Error generating district dashboard: {e}")