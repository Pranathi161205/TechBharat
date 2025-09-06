# main.py
import os
import pandas as pd
import yaml
from scripts.clean_data import clean_data, load_data
from scripts.transform_data import transform_data
from scripts.analyze_data import analyze_data, find_anomalies, generate_executive_summary, run_root_cause_analysis
from scripts.predict_data import predict_future_kits, predict_high_risk
from scripts.visualize_data import create_choropleth_map
from scripts.generate_report import generate_html_report
from scripts.dashboard import create_dashboard, create_district_dashboard

# Define file paths
DATA_DIR = 'data'
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'data_cleaned.csv')
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'data_transformed.csv')

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
    This version resolves configured column names to actual dataframe columns
    (handles 'District' vs 'district' vs 'districtName' mismatches).
    """
    import re
    def _normalize(s):
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = s.replace('\u00b0', 'deg').replace('°', 'deg').replace('Â', '')
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"[\s\-]+", "_", s)
        return s

    def resolve_column(df, configured_name, preferred_candidates=None):
        """Return actual column name in df that best matches configured_name or preferred candidates."""
        cols = list(df.columns)
        # 1) If configured_name exactly exists, use it
        if configured_name and configured_name in cols:
            return configured_name
        # 2) If logical candidates exist in df, prefer them
        if preferred_candidates:
            for cand in preferred_candidates:
                if cand in cols:
                    return cand
        # 3) Try normalized exact match
        if configured_name:
            nc = _normalize(configured_name)
            for c in cols:
                if _normalize(c) == nc:
                    return c
        # 4) Try normalized candidate matches
        for c in cols:
            nc = _normalize(c)
            for cand in (preferred_candidates or []):
                if _normalize(cand) == nc:
                    return c
        # 5) Try token overlap / substring: prefer columns that contain the token parts of configured_name
        if configured_name:
            tokens = set(_normalize(configured_name).split('_'))
            best, best_score = None, 0
            for c in cols:
                score = len(tokens.intersection(set(_normalize(c).split('_'))))
                if score > best_score:
                    best_score = score
                    best = c
            if best_score > 0:
                return best
        # not found
        return None

    global transformed_df, kit_threshold, anc_threshold, high_risk_threshold, anc2_threshold, cleaned_df_global, config, current_dataset_name

    if transformed_df is None:
        try:
            transformed_df = pd.read_csv(TRANSFORMED_DATA_PATH)
            print("Interactive mode started. Transformed data loaded.")
        except FileNotFoundError:
            print("Error: Transformed data not found. Please run the full pipeline first.")
            return

    # Resolve district column ONCE for this session
    configured_district = None
    try:
        configured_district = config[current_dataset_name]['columns'].get('district', None)
    except Exception:
        configured_district = None

    # prefer these logical names if present
    preferred = ['district', 'districtName', 'district_name', 'districtname']
    district_col = resolve_column(transformed_df, configured_district, preferred_candidates=preferred)
    if district_col is None:
        print("Warning: Could not resolve district column automatically. Some commands may fail until you set dataset/config correctly.")
    else:
        # normalize district values to title-case for lookups, but keep column as-is
        try:
            transformed_df[district_col] = transformed_df[district_col].astype(str).str.strip().str.title()
        except Exception:
            pass

    while True:
        command = input("\nRTGS-CLI> ").strip()

        if not command:
            continue

        cmd_lower = command.lower()

        if cmd_lower in ['exit', 'quit']:
            print("Exiting interactive mode. Goodbye!")
            break

        elif cmd_lower.startswith('get_insights '):
            parts = command.split(' ')
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metric_name = parts[2].strip()
                
                # Check if metric exists
                if metric_name not in transformed_df.columns:
                    print(f"Error: Metric '{metric_name}' not found. Please check spelling.")
                    continue

                if district_col is None:
                    print("Error: District column not resolved. Cannot perform district-level lookup.")
                    continue

                district_insights = transformed_df[transformed_df[district_col] == district_name]
                
                if not district_insights.empty:
                    print(district_insights[[district_col, metric_name]].to_string(index=False))
                else:
                    print(f"No data found for district '{district_name}'. Please check the spelling.")
            else:
                print("Invalid command. Usage: get_insights <district_name> <metric_name>")

        # --- Thresholds ---
        elif cmd_lower.startswith('set_threshold '):
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

        # --- Run analysis ---
        elif cmd_lower == 'run_analysis':
            print("Running full analysis with current thresholds...")
            insights = analyze_data(transformed_df, config[current_dataset_name])
            if insights:
                print(insights)

        # --- Find anomalies ---
        elif cmd_lower.startswith('find_anomalies '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metric = parts[1].strip()
                metrics_to_check = config[current_dataset_name].get('metrics_to_calculate', [])
                if metric in metrics_to_check:
                    anomalies, message = find_anomalies(transformed_df.copy(), metric)
                    print(message)
                    if anomalies is not None and not anomalies.empty:
                        # try to resolve district column again if necessary
                        dcol = district_col or resolve_column(transformed_df, configured_district, preferred_candidates=preferred)
                        if dcol:
                            print(anomalies[[dcol, metric, 'z_score']].to_string(index=False))
                        else:
                            print(anomalies.to_string(index=False))
                else:
                    print("Invalid metric. Please choose from metrics in config.")
            else:
                print("Invalid command. Usage: find_anomalies <metric_name>")

        # --- Predict ---
        elif cmd_lower.startswith('predict '):
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

        # --- Root cause analysis ---
        elif cmd_lower.startswith('root_cause '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                problem_metric = parts[1].strip()
                if problem_metric in transformed_df.columns:
                    print(f"Running root cause analysis for '{problem_metric}'...")
                    correlations = run_root_cause_analysis(transformed_df.copy(), problem_metric)
                    print(correlations)
                else:
                    print(f"Error: Metric '{problem_metric}' not found. Please check spelling.")
            else:
                print("Invalid command. Usage: root_cause <metric_name>")

        # --- Dashboard ---
        elif cmd_lower.startswith('generate_dashboard '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metrics = [m.strip() for m in parts[1].split(',')]
                print(f"Generating dashboard for metrics: {metrics}")
                try:
                    create_dashboard(transformed_df.copy(), metrics)
                except Exception as e:
                    print(f"   - Error: {e}")
            else:
                print("Invalid command. Usage: generate_dashboard <metric1,metric2,etc.>")

        elif cmd_lower.startswith('dashboard_for '):
            parts = command.split(' ', 2)
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metrics = [m.strip() for m in parts[2].split(',')]
                output_filename = f"data/{district_name.replace(' ', '_')}_dashboard.png"
                print(f"Generating dashboard for {district_name} with metrics: {metrics}")
                try:
                    create_district_dashboard(transformed_df.copy(), district_name, metrics, output_filename)
                except Exception as e:
                    print(f"   - Error: {e}")
            else:
                print("Invalid command. Usage: dashboard_for <district_name> <metric1,metric2,etc.>")

        # --- Set dataset ---
        elif cmd_lower.startswith('set_dataset '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                dataset_name = parts[1].strip()
                if dataset_name in config:
                    current_dataset_name = dataset_name
                    print(f"Dataset set to '{current_dataset_name}'. Please run the full pipeline to load new data.")
                    run_pipeline_and_start_cli()
                    break
                else:
                    print(f"Error: Dataset '{dataset_name}' not found in config.")
            else:
                print("Invalid command. Usage: set_dataset <dataset_name>")

        elif cmd_lower == 'help':
            print("\nAvailable commands:")
            print("  get_insights <district_name> <metric_name> - Get a specific metric for a district.")
            print("  set_threshold <metric> <value> - Set a new threshold. Metrics: kits, anc, high_risk.")
            print("  run_analysis                - Re-run the full analysis with current thresholds.")
            print("  find_anomalies <metric>      - Find statistical outliers (anomalies) in a metric.")
            print("  predict <metric>            - Get a prediction for 'kits' or 'high_risk'.")
            print("  generate_dashboard <metric1,metric2,...> - Generate a dashboard for specified metrics.")
            print("  dashboard_for <district_name> <metric1,metric2,...> - Generate a dashboard for a specific district.")
            print("  root_cause <metric_name>    - Suggests potential causes for a problem metric.")
            print("  set_dataset <dataset_name>  - Switch to a different dataset from the config file.")
            print("  exit / quit                   - Exit the interactive mode.")
        else:
            print("Unknown command. Type 'help' for a list of commands.")


def run_pipeline_and_start_cli():
    """
    Runs the full pipeline, then starts the interactive CLI.
    """
    global transformed_df, cleaned_df_global, config, current_dataset_name
    
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Error: config.yaml not found. Aborting.")
        return

    print("[Policymaker CLI] --> [RTGS Agent]")
    print(f"Using dataset: {current_dataset_name}")
    
    dataset_config = config[current_dataset_name]
    file_path_from_config = dataset_config['file_path']
    columns_from_config = dataset_config['columns']

    print(f"1. Loading {current_dataset_name} data...")
    raw_df = load_data(file_path_from_config)
    if raw_df is None: return
    
    print(f"2. Cleaning & Standardizing {current_dataset_name} data...")
    cleaned_df = clean_data(raw_df, columns_from_config)
    if cleaned_df is None: return
    cleaned_df.to_csv(os.path.join(DATA_DIR, f'{current_dataset_name}_cleaned.csv'), index=False)
    cleaned_df_global = cleaned_df
    
    print("\n3. Transforming Data...")
    transformed_df = transform_data(cleaned_df, dataset_config, current_dataset_name)
    if transformed_df is None: return
    transformed_df.to_csv(os.path.join(DATA_DIR, f'{current_dataset_name}_transformed.csv'), index=False)
    
    print("\n4. Analyzing Data & Generating Initial Insights...")
    insights = analyze_data(transformed_df, dataset_config)
    if insights:
        print("\n\n5. Outputting Initial Insights:")
        print(insights)

    print("\n5b. Outputting Executive Summary for Key Districts:")
    if current_dataset_name == 'health_data':
        key_districts = ['Hyderabad', 'Medchal-Malkajgiri', 'Karimnagar']
        executive_summary = generate_executive_summary(transformed_df, key_districts)
        print(executive_summary)
    else:
        print("   - Executive summary is only available for health data.")

    print("\n6. Running Predictive Analysis...")
    try:
        if current_dataset_name == 'health_data':
            predicted_kits, _ = predict_future_kits(cleaned_df.copy())
            predicted_high_risk, prediction_date = predict_high_risk(cleaned_df.copy())
            print(f"   - Predicted MCH kits for {prediction_date}: {predicted_kits}")
            print(f"   - Predicted high-risk pregnancies for {prediction_date}: {predicted_high_risk}")
        else:
            print("   - Predictive analysis is only available for health data.")
    except Exception as e:
        print(f"   - Error during prediction: {e}")
        
    print("\n7. Creating Geospatial Visualization...")
    try:
        if current_dataset_name == 'health_data':
            create_choropleth_map(transformed_df)
        else:
            print("   - Geospatial visualization is only available for health data.")
    except Exception as e:
        print(f"   - Error during visualization: {e}")
        
    print("\n8. Generating Comprehensive HTML Report...")
    print("   - Generating report is currently not data agnostic.")
    
    print("\n9. Creating Dashboard Visualization...")
    try:
        if current_dataset_name == 'health_data':
            default_metrics = ['kit_coverage_ratio', 'high_risk_ratio']
            create_dashboard(transformed_df.copy(), default_metrics)
        else:
            print("   - Dashboard creation is currently not data agnostic.")
    except Exception as e:
        print(f"   - Error during dashboard creation: {e}")

    print("\n------------------------------------------------")
    print("Pipeline Complete. Starting Interactive CLI.")
    run_interactive_mode()


if __name__ == '__main__':
    run_pipeline_and_start_cli()
