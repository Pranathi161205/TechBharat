# main.py

import os
import pandas as pd
from scripts.clean_data import clean_data, load_data
from scripts.transform_data import transform_data
from scripts.analyze_data import analyze_data, find_anomalies
from scripts.predict_data import predict_future_kits
from scripts.visualize_data import create_choropleth_map

# Define file paths
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_cleaned.csv')
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_transformed.csv')

# Placeholder for the main dataframe
transformed_df = None
cleaned_df_global = None

# Define global variables for thresholds
kit_threshold = 0.8
anc_threshold = 0.5
high_risk_threshold = 0.1

def run_interactive_mode():
    """
    Enters an interactive loop to handle user commands.
    """
    global transformed_df, kit_threshold, anc_threshold, high_risk_threshold, cleaned_df_global
    
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
            parts = command.split(' ', 1)
            if len(parts) > 1:
                district_name = parts[1].strip().title()
                print(f"Generating insights for {district_name}...")
                
                district_insights = transformed_df[transformed_df['districtName'] == district_name]
                
                if not district_insights.empty:
                    print(district_insights[['districtName', 'kit_coverage_ratio', 'high_risk_ratio', 'anc4_to_anc1_ratio']].to_string(index=False))
                else:
                    print(f"No data found for district '{district_name}'. Please check the spelling.")
            else:
                print("Invalid command. Usage: get_insights <district_name>")
        
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
                    elif metric == 'high_risk':
                        high_risk_threshold = value
                        print(f"High-risk ratio threshold set to {high_risk_threshold*100}%.")
                    else:
                        print("Invalid metric. Options are: kits, anc, high_risk")
                except ValueError:
                    print("Invalid value. Please enter a number.")
            else:
                print("Invalid command. Usage: set_threshold <metric> <value>")
        
        elif command == 'run_analysis':
            print("Running full analysis with current thresholds...")
            insights = analyze_data(transformed_df, kit_threshold=kit_threshold, anc_threshold=anc_threshold, high_risk_threshold=high_risk_threshold)
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
                target_date = parts[1].strip()
                try:
                    predicted_kits, prediction_date = predict_future_kits(cleaned_df_global.copy(), target_date)
                    print(f"\n- Predicted MCH kits for {prediction_date}: {predicted_kits}")
                except Exception as e:
                    print(f"   - Error during prediction: {e}. Please enter a valid date format (e.g., '2024-03-01').")
            else:
                print("Invalid command. Usage: predict <YYYY-MM-DD>")
        
        elif command == 'help':
            print("\nAvailable commands:")
            print("  get_insights <district_name>  - Get key metrics for a specific district.")
            print("  set_threshold <metric> <value> - Set a new threshold. Metrics: kits, anc, high_risk.")
            print("  run_analysis                - Re-run the full analysis with current thresholds.")
            print("  find_anomalies <metric>      - Find statistical outliers (anomalies) in a metric.")
            print("  predict <YYYY-MM-DD>        - Get a prediction for a specific date.")
            print("  exit / quit                   - Exit the interactive mode.")
        else:
            print("Unknown command. Type 'help' for a list of commands.")

def run_pipeline_and_start_cli():
    """
    Runs the full pipeline, then starts the interactive CLI.
    """
    global transformed_df, cleaned_df_global
    
    print("[Policymaker CLI] --> [RTGS Agent]")
    print("1. Loading Health Dataset...")
    raw_df = load_data(RAW_DATA_PATH)
    if raw_df is None:
        print("Pipeline aborted due to file loading error.")
        return

    print("2. Cleaning & Standardizing Data...")
    cleaned_df = clean_data(raw_df)
    if cleaned_df is None:
        return
    print(f"   - Original rows: {len(raw_df)}")
    print(f"   - Cleaned rows: {len(cleaned_df)}")
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"   - Cleaned data saved to {CLEANED_DATA_PATH}")
    cleaned_df_global = cleaned_df
    
    print("\n3. Transforming Data...")
    transformed_df = transform_data(cleaned_df)
    if transformed_df is None:
        return
    transformed_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    print(f"   - Transformed data saved to {TRANSFORMED_DATA_PATH}")
    
    print("\n4. Analyzing Data & Generating Initial Insights...")
    insights = analyze_data(transformed_df)
    if insights:
        print("\n\n5. Outputting Initial Insights:")
        print(insights)

    print("\n6. Running Predictive Analysis...")
    try:
        predicted_kits, prediction_date = predict_future_kits(cleaned_df.copy())
        print(f"   - Predicted MCH kits for {prediction_date}: {predicted_kits}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")
        
    print("\n7. Creating Geospatial Visualization...")
    try:
        create_choropleth_map(transformed_df)
    except Exception as e:
        print(f"   - Error during visualization: {e}")
        
    print("\n------------------------------------------------")
    print("Pipeline Complete. Starting Interactive CLI.")
    run_interactive_mode()

    # main.py

# ... (all your existing imports) ...
from scripts.generate_report import generate_html_report # Import the new function

# ... (all your existing global variables and functions) ...

def run_pipeline_and_start_cli():
    # ... (all your existing code for loading, cleaning, transforming) ...
    global transformed_df, cleaned_df_global
    
    print("[Policymaker CLI] --> [RTGS Agent]")
    print("1. Loading Health Dataset...")
    raw_df = load_data(RAW_DATA_PATH)
    if raw_df is None: return
    cleaned_df = clean_data(raw_df)
    if cleaned_df is None: return
    cleaned_df_global = cleaned_df
    
    print("\n3. Transforming Data...")
    transformed_df = transform_data(cleaned_df)
    if transformed_df is None: return

    # ... (rest of your existing code up to this point) ...
    
    # Run the initial analysis to capture the insights log
    print("\n4. Analyzing Data & Generating Initial Insights...")
    insights = analyze_data(transformed_df)
    if insights:
        print("\n\n5. Outputting Initial Insights:")
        print(insights)

    # Run the predictive analysis to get the prediction
    print("\n6. Running Predictive Analysis...")
    try:
        predicted_kits, prediction_date = predict_future_kits(cleaned_df.copy())
        print(f"   - Predicted MCH kits for {prediction_date}: {predicted_kits}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")
        predicted_kits, prediction_date = "N/A", "N/A"
    
    # Run the visualization step to create the map file first
    print("\n7. Creating Geospatial Visualization...")
    try:
        create_choropleth_map(transformed_df)
    except Exception as e:
        print(f"   - Error during visualization: {e}")

    # NEW: Generate the full HTML report with all the data
    print("\n8. Generating Comprehensive HTML Report...")
    generate_html_report(transformed_df, insights, predicted_kits, prediction_date)

    print("\n------------------------------------------------")
    print("Pipeline Complete. Starting Interactive CLI.")
    run_interactive_mode()



if __name__ == '__main__':
    run_pipeline_and_start_cli()