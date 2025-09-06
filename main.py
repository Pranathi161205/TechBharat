# main.py

import os
import pandas as pd
import yaml # You'll need to install this library: pip install PyYAML
from scripts.clean_data import clean_data, load_data
from scripts.transform_data import transform_data
from scripts.analyze_data import analyze_data, find_anomalies, generate_executive_summary, run_root_cause_analysis
from scripts.predict_data import predict_future_kits, predict_high_risk
from scripts.visualize_data import create_choropleth_map
from scripts.generate_report import generate_html_report
from scripts.dashboard import create_dashboard, create_district_dashboard

# Define file paths
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_cleaned.csv')
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_transformed.csv')

# Global config variable
config = None
current_dataset_name = 'health_data'

# Placeholder for the main dataframe
transformed_df = None
cleaned_df_global = None

# Define global variables for thresholds
kit_threshold = 0.8
anc_threshold = 0.5
anc2_threshold = 0.9
high_risk_threshold = 0.1

def run_interactive_mode():
    """
    Enters an interactive loop to handle user commands.
    """
    global transformed_df, kit_threshold, anc_threshold, high_risk_threshold, anc2_threshold, cleaned_df_global, config, current_dataset_name
    
    if transformed_df is None:
        try:
            transformed_df = pd.read_csv(TRANSFORMED_DATA_PATH)
            print("Interactive mode started. Transformed data loaded.")
        except FileNotFoundError:
            print("Error: Transformed data not found. Please run the full pipeline first.")
            return

    while True:
        command = input("\nRTGS-CLI> ").strip().lower()

        if command == 'exit' or command == 'quit':
            print("Exiting interactive mode. Goodbye!")
            break
        elif command.startswith('get_insights '):
            parts = command.split(' ')
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metric_name = parts[2].strip().lower()
                
                if metric_name not in transformed_df.columns:
                    print(f"Error: Metric '{metric_name}' not found. Please check spelling.")
                    continue
                
                print(f"Generating insights for {district_name} for metric '{metric_name}'...")
                
                district_insights = transformed_df[transformed_df['districtName'] == district_name]
                
                if not district_insights.empty:
                    print(district_insights[['districtName', metric_name]].to_string(index=False))
                else:
                    print(f"No data found for district '{district_name}'. Please check the spelling.")
            else:
                print("Invalid command. Usage: get_insights <district_name> <metric_name>")
        
        elif command.startswith('set_threshold '):
            parts = command.split(' ')
            if len(parts) == 3:
                metric = parts[1].lower()
                try:
                    value = float(parts[2])
                    if metric == 'kits':
                        kit_threshold = value
                        print(f"Kit coverage threshold set to {kit_threshold*100}%.")
                    elif metric == 'anc':
                        anc_threshold = value
                        print(f"ANC completion threshold set to {anc_threshold*100}%.")
                    elif metric == 'anc2':
                        anc2_threshold = value
                        print(f"ANC2-to-ANC1 follow-up threshold set to {anc2_threshold*100}%.")
                    elif metric == 'high_risk':
                        high_risk_threshold = value
                        print(f"High-risk ratio threshold set to {high_risk_threshold*100}%.")
                    else:
                        print("Invalid metric. Options are: kits, anc, anc2, high_risk")
                except ValueError:
                    print("Invalid value. Please enter a number.")
            else:
                print("Invalid command. Usage: set_threshold <metric> <value>")
        
        elif command == 'run_analysis':
            print("Running full analysis with current thresholds...")
            insights = analyze_data(transformed_df, kit_threshold=kit_threshold, anc_threshold=anc_threshold, anc2_threshold=anc2_threshold, high_risk_threshold=high_risk_threshold)
            if insights:
                print(insights)
        
        elif command.startswith('find_anomalies '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metric = parts[1].strip().lower()
                if metric in ['total_kits_distributed', 'total_high_risk']:
                    anomalies, message = find_anomalies(transformed_df.copy(), metric)
                    print(message)
                    if anomalies is not None and not anomalies.empty:
                        print(anomalies[['districtName', metric, 'z_score']].to_string(index=False))
                else:
                    print("Invalid metric. Please choose 'total_kits_distributed' or 'total_high_risk'.")
            else:
                print("Invalid command. Usage: find_anomalies <metric_name>")

        elif command.startswith('predict '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                prediction_metric = parts[1].strip().lower()
                if prediction_metric == 'kits':
                    predicted_kits, prediction_date = predict_future_kits(cleaned_df_global.copy())
                    print(f"\n- Predicted MCH kits for {prediction_date}: {predicted_kits}")
                elif prediction_metric == 'high_risk':
                    predicted_high_risk, prediction_date = predict_high_risk(cleaned_df_global.copy())
                    print(f"\n- Predicted high-risk pregnancies for {prediction_date}: {predicted_high_risk}")
                else:
                    print("Invalid metric. Please choose 'kits' or 'high_risk'.")
            else:
                print("Invalid command. Usage: predict <metric>")
        
        elif command.startswith('root_cause '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                problem_metric = parts[1].strip().lower()
                if problem_metric in transformed_df.columns:
                    print(f"Running root cause analysis for '{problem_metric}'...")
                    correlations = run_root_cause_analysis(transformed_df.copy(), problem_metric)
                    print(correlations)
                else:
                    print(f"Error: Metric '{problem_metric}' not found. Please check spelling.")
            else:
                print("Invalid command. Usage: root_cause <metric_name>")

        elif command.startswith('generate_dashboard '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metrics = parts[1].strip().split(',')
                print(f"Generating dashboard for metrics: {metrics}")
                try:
                    create_dashboard(transformed_df.copy(), metrics)
                except Exception as e:
                    print(f"   - Error: {e}")
            else:
                print("Invalid command. Usage: generate_dashboard <metric1,metric2,etc.>")
        
        elif command.startswith('dashboard_for '):
            parts = command.split(' ', 2)
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metrics = parts[2].strip().split(',')
                output_filename = f"data/{district_name.replace(' ', '_')}_dashboard.png"
                print(f"Generating dashboard for {district_name} with metrics: {metrics}")
                try:
                    create_district_dashboard(transformed_df.copy(), district_name, metrics, output_filename)
                except Exception as e:
                    print(f"   - Error: {e}")
            else:
                print("Invalid command. Usage: dashboard_for <district_name> <metric1,metric2,etc.>")

        elif command == 'help':
            print("\nAvailable commands:")
            print("  get_insights <district_name> <metric_name> - Get a specific metric for a district.")
            print("  set_threshold <metric> <value> - Set a new threshold. Metrics: kits, anc, high_risk.")
            print("  run_analysis                - Re-run the full analysis with current thresholds.")
            print("  find_anomalies <metric>      - Find statistical outliers (anomalies) in a metric.")
            print("  predict <metric>            - Get a prediction for 'kits' or 'high_risk'.")
            print("  generate_dashboard <metric1,metric2,...> - Generate a dashboard for specified metrics.")
            print("  dashboard_for <district_name> <metric1,metric2,...> - Generate a dashboard for a specific district.")
            print("  root_cause <metric_name>    - Suggests potential causes for a problem metric.")
            print("  exit / quit                   - Exit the interactive mode.")
        else:
            print("Unknown command. Type 'help' for a list of commands.")

def run_pipeline_and_start_cli():
    """
    Runs the full pipeline, then starts the interactive CLI.
    """
    global transformed_df, cleaned_df_global, config, current_dataset_name
    
    # Load the config file at startup
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Error: config.yaml not found. Aborting.")
        return

    print("[Policymaker CLI] --> [RTGS Agent]")
    print(f"Using dataset: {current_dataset_name}")
    
    # Get the file path from the config
    file_path_from_config = config[current_dataset_name]['file_path']

    # Step 1: Load Health Dataset from config
    print("1. Loading Health Dataset...")
    raw_df = load_data(file_path_from_config)
    if raw_df is None: return
    
    print("2. Cleaning & Standardizing Data...")
    cleaned_df = clean_data(raw_df)
    if cleaned_df is None: return
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    cleaned_df_global = cleaned_df
    
    print("\n3. Transforming Data...")
    transformed_df = transform_data(cleaned_df)
    if transformed_df is None: return
    transformed_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    
    print("\n4. Analyzing Data & Generating Initial Insights...")
    insights = analyze_data(transformed_df)
    if insights:
        print("\n\n5. Outputting Initial Insights:")
        print(insights)

    print("\n5b. Outputting Executive Summary for Key Districts:")
    key_districts = ['Hyderabad', 'Medchal-Malkajgiri', 'Karimnagar']
    executive_summary = generate_executive_summary(transformed_df, key_districts)
    print(executive_summary)

    print("\n6. Running Predictive Analysis...")
    try:
        predicted_kits, _ = predict_future_kits(cleaned_df.copy())
        predicted_high_risk, prediction_date = predict_high_risk(cleaned_df.copy())
        print(f"   - Predicted MCH kits for {prediction_date}: {predicted_kits}")
        print(f"   - Predicted high-risk pregnancies for {prediction_date}: {predicted_high_risk}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")
        predicted_kits, predicted_high_risk, prediction_date = "N/A", "N/A", "N/A"
        
    print("\n7. Creating Geospatial Visualization...")
    try:
        create_choropleth_map(transformed_df)
    except Exception as e:
        print(f"   - Error during visualization: {e}")
        
    print("\n8. Generating Comprehensive HTML Report...")
    generate_html_report(transformed_df, insights, predicted_kits, predicted_high_risk, prediction_date)

    print("\n9. Creating Dashboard Visualization...")
    try:
        default_metrics = ['kit_coverage_ratio', 'high_risk_ratio']
        create_dashboard(transformed_df.copy(), default_metrics)
    except Exception as e:
        print(f"   - Error during dashboard creation: {e}")

    print("\n------------------------------------------------")
    print("Pipeline Complete. Starting Interactive CLI.")
    run_interactive_mode()

if __name__ == '__main__':
    run_pipeline_and_start_cli()