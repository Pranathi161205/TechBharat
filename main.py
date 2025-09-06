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
import difflib

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

# -----------------------
# Helper utilities
# -----------------------
def _normalize_for_lookup(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace('\u00b0', 'deg').replace('°', 'deg').replace('Â', '')
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == '_')
    s = "_".join(s.split())
    return s

def resolve_column_name_from_config_or_df(df: pd.DataFrame, dataset_cfg: dict, logical_name: str, preferred: list = None):
    """
    Try to resolve a real column name in df for the logical_name using:
      1) dataset_cfg['columns'][logical_name] if provided,
      2) preferred candidate names,
      3) normalized matching against df columns.
    Returns actual df column name or None.
    """
    # 1) config mapping
    try:
        if dataset_cfg and 'columns' in dataset_cfg and logical_name in dataset_cfg['columns']:
            candidate = dataset_cfg['columns'][logical_name]
            if candidate in df.columns:
                return candidate
            # try normalized match
            nc = _normalize_for_lookup(candidate)
            for c in df.columns:
                if _normalize_for_lookup(c) == nc:
                    return c
    except Exception:
        pass

    # 2) preferred candidates
    if preferred:
        for cand in preferred:
            if cand in df.columns:
                return cand

    # 3) normalized match to logical name
    nc = _normalize_for_lookup(logical_name)
    for c in df.columns:
        if _normalize_for_lookup(c) == nc:
            return c

    # 4) token / substring match
    for c in df.columns:
        if nc in _normalize_for_lookup(c) or _normalize_for_lookup(c) in nc:
            return c

    return None

# -----------------------
# Pipeline runner (reusable)
# -----------------------
def run_pipeline(dataset_name: str, interactive: bool = False):
    """
    Runs the full pipeline for the given dataset name in config.
    Returns (cleaned_df, transformed_df) or (None, None) on error.
    If interactive is True the function will not block (same behavior as old pipeline).
    """
    global transformed_df, cleaned_df_global, config, current_dataset_name

    if config is None:
        try:
            with open('config.yaml', 'r') as file:
                cfg = yaml.safe_load(file)
                globals()['config'] = cfg
        except FileNotFoundError:
            print("Error: config.yaml not found. Aborting pipeline.")
            return None, None

    if dataset_name not in config:
        print(f"Error: dataset '{dataset_name}' not found in config.")
        return None, None

    dataset_config = config[dataset_name]
    file_path = dataset_config.get('file_path')
    if not file_path:
        print(f"Error: 'file_path' not defined for dataset '{dataset_name}' in config.yaml.")
        return None, None

    print(f"[Pipeline] 1. Loading {dataset_name} data...")
    raw_df = load_data(file_path)
    if raw_df is None:
        return None, None

    print(f"[Pipeline] 2. Cleaning & Standardizing {dataset_name} data...")
    cleaned_df = clean_data(raw_df, dataset_config.get('columns', {}))
    if cleaned_df is None:
        return None, None

    # ensure data directory exists for outputs
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save cleaned
    cleaned_out = os.path.join(DATA_DIR, f"{dataset_name}_cleaned.csv")
    cleaned_df.to_csv(cleaned_out, index=False)
    cleaned_df_global = cleaned_df

    print(f"[Pipeline] 3. Transforming {dataset_name} data...")
    transformed = transform_data(cleaned_df, dataset_config, dataset_name)
    if transformed is None:
        return cleaned_df, None

    transformed_out = os.path.join(DATA_DIR, f"{dataset_name}_transformed.csv")
    transformed.to_csv(transformed_out, index=False)

    # set globals
    transformed_df = transformed

    print(f"[Pipeline] 4. Analyzing data & generating initial insights...")
    try:
        insights = analyze_data(transformed, dataset_config)
        if insights:
            print("\nInitial Insights:")
            print(insights)
    except Exception as e:
        print(f"   - Error during analysis: {e}")

    # Executive summary for health_data
    if dataset_name == 'health_data':
        try:
            key_districts = ['Hyderabad', 'Medchal-Malkajgiri', 'Karimnagar']
            executive_summary = generate_executive_summary(transformed, key_districts)
            print("\nExecutive summary for key districts:")
            print(executive_summary)
        except Exception as e:
            print(f"   - Error generating executive summary: {e}")

    # -----------------------
    # Additional dataset-specific insights (tourism_domestic)
    # -----------------------
    if dataset_name == "tourism_domestic":
        try:
            tdf = transformed.copy()

            # Top 5 districts by total visitors
            if 'total_visitors' in tdf.columns:
                top5 = tdf.sort_values('total_visitors', ascending=False).head(5)
                print("\nTop 5 districts by total visitors (2024):")
                for i, row in enumerate(top5[['district', 'total_visitors']].to_dict(orient='records'), start=1):
                    print(f"  {i}. {row['district']}: {row['total_visitors']:,}")
            else:
                print("\n(total_visitors not found in transformed tourism dataset.)")

            # Top 5 by avg monthly visitors
            if 'avg_monthly_visitors' in tdf.columns:
                top_avg = tdf.sort_values('avg_monthly_visitors', ascending=False).head(5)
                print("\nTop 5 districts by average monthly visitors:")
                for i, r in enumerate(top_avg[['district','avg_monthly_visitors']].to_dict(orient='records'), start=1):
                    print(f"  {i}. {r['district']}: {r['avg_monthly_visitors']:,}")

            # Seasonality: most common peak months
            if 'peak_month' in tdf.columns:
                peaks = tdf['peak_month'].fillna('Unknown')
                peak_counts = peaks.value_counts().head(5)
                if not peak_counts.empty:
                    print("\nMost frequent peak months across districts:")
                    for month, cnt in peak_counts.items():
                        print(f"  {month}: {cnt} districts")

            # Save lightweight insights CSV
            insights_out = os.path.join(DATA_DIR, f"{dataset_name}_insights.csv")
            tdf.sort_values('total_visitors', ascending=False).to_csv(insights_out, index=False)
            print(f"\nTourism insights saved to: {insights_out}")

            # Create dashboard using helper or fallback to a simple plot
            try:
                create_dashboard(tdf.copy(), ['total_visitors', 'avg_monthly_visitors'])
            except Exception:
                try:
                    import matplotlib.pyplot as plt
                    out_img = os.path.join(DATA_DIR, f"{dataset_name}_top20.png")
                    plot_df = tdf.sort_values('total_visitors', ascending=False).head(20)
                    plt.figure(figsize=(10,8))
                    plt.barh(plot_df['district'].astype(str), plot_df['total_visitors'])
                    plt.gca().invert_yaxis()
                    plt.xlabel('Total Visitors (2024)')
                    plt.title('Top 20 districts by domestic visitors (2024)')
                    plt.tight_layout()
                    plt.savefig(out_img, dpi=150)
                    plt.close()
                    print(f"Tourism dashboard image saved to: {out_img}")
                except Exception as e:
                    print("Could not create tourism dashboard image:", e)
        except Exception as e:
            print("Error creating tourism-specific insights:", e)

    # Predictive analysis
    print("\n[Pipeline] 5. Running predictive analysis (if available)...")
    try:
        if dataset_name == 'health_data':
            predicted_kits, _ = predict_future_kits(cleaned_df.copy())
            predicted_high_risk, prediction_date = predict_high_risk(cleaned_df.copy())
            print(f"   - Predicted MCH kits for {prediction_date}: {predicted_kits}")
            print(f"   - Predicted high-risk pregnancies for {prediction_date}: {predicted_high_risk}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")

    # Visualization
    print("\n[Pipeline] 6. Creating geospatial visualization (if available)...")
    try:
        if dataset_name == 'health_data':
            create_choropleth_map(transformed)
        else:
            print("   - Geospatial visualization not configured for this dataset.")
    except Exception as e:
        print(f"   - Error during visualization: {e}")

    # Dashboard (default for health_data)
    print("\n[Pipeline] 7. Creating dashboard visualization (if available)...")
    try:
        if dataset_name == 'health_data':
            default_metrics = ['kit_coverage_ratio', 'high_risk_ratio']
            create_dashboard(transformed.copy(), default_metrics)
    except Exception as e:
        print(f"   - Error during dashboard creation: {e}")

    print("\n[Pipeline] Completed for dataset:", dataset_name)
    # If interactive flag is False, return without starting CLI
    return cleaned_df, transformed

# -----------------------
# Interactive CLI
# -----------------------
def run_interactive_mode():
    """
    Command-line interactive loop for the policymaker CLI.
    Resolves district column names robustly for lookups.
    """
    import re

    def resolve_column(df, configured_name, preferred_candidates=None):
        # tries a few heuristics to find the real column name
        cols = list(df.columns)
        # exact
        if configured_name and configured_name in cols:
            return configured_name
        if preferred_candidates:
            for cand in preferred_candidates:
                if cand in cols:
                    return cand
        if configured_name:
            nc = _normalize_for_lookup(configured_name)
            for c in cols:
                if _normalize_for_lookup(c) == nc:
                    return c
        # normalized candidate matches
        for c in cols:
            nc = _normalize_for_lookup(c)
            if preferred_candidates:
                for cand in preferred_candidates:
                    if _normalize_for_lookup(cand) == nc:
                        return c
        # token overlap
        if configured_name:
            target_tokens = set(_normalize_for_lookup(configured_name).split('_'))
            best, best_score = None, 0
            for c in cols:
                score = len(target_tokens.intersection(set(_normalize_for_lookup(c).split('_'))))
                if score > best_score:
                    best_score = score
                    best = c
            if best_score > 0:
                return best
        return None

    global transformed_df, cleaned_df_global, config, current_dataset_name, kit_threshold, anc_threshold, anc2_threshold, high_risk_threshold

    if transformed_df is None:
        try:
            transformed_df = pd.read_csv(TRANSFORMED_DATA_PATH)
            print("Interactive mode started. Transformed data loaded.")
        except FileNotFoundError:
            print("Error: Transformed data not found. Please run the pipeline first.")
            return

    # pre-resolve district column for convenience
    configured_district = None
    try:
        configured_district = config[current_dataset_name]['columns'].get('district', None)
    except Exception:
        configured_district = None

    preferred = ['district', 'districtName', 'district_name', 'districtname', 'District']
    district_col = resolve_column(transformed_df, configured_district, preferred_candidates=preferred)
    if district_col:
        try:
            transformed_df[district_col] = transformed_df[district_col].astype(str).str.strip().str.title()
        except Exception:
            pass
    else:
        print("Warning: could not resolve district column automatically. Some commands may not work.")

    # CLI loop
    while True:
        command = input("\nRTGS-CLI> ").strip()
        if not command:
            continue
        cmd = command.lower()

        if cmd in ['exit', 'quit']:
            print("Exiting interactive mode. Goodbye!")
            break

        if cmd == 'list_metrics':
            try:
                print("Available metrics / columns:")
                print(", ".join([str(c) for c in transformed_df.columns.tolist()]))
            except Exception:
                print("Could not list metrics.")

        elif cmd.startswith('get_insights '):
            parts = command.split(' ')
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metric_name = parts[2].strip()
                if metric_name not in transformed_df.columns:
                    # try fuzzy match for metric name
                    candidates = transformed_df.columns.tolist()
                    match = difflib.get_close_matches(metric_name, candidates, n=1, cutoff=0.6)
                    if match:
                        metric_name = match[0]
                        print(f"Using metric '{metric_name}' (matched from input).")
                    else:
                        print(f"Error: Metric '{metric_name}' not found.")
                        continue
                if district_col is None:
                    print("Error: District column not resolved. Cannot lookup by district.")
                    continue
                result = transformed_df[transformed_df[district_col] == district_name]
                if not result.empty:
                    print(result[[district_col, metric_name]].to_string(index=False))
                else:
                    print(f"No data for district '{district_name}'.")
            else:
                print("Invalid usage. get_insights <district_name> <metric_name>")

        elif cmd.startswith('set_threshold '):
            parts = command.split(' ')
            if len(parts) == 3:
                metric = parts[1].lower()
                try:
                    value = float(parts[2])
                except ValueError:
                    print("Invalid number.")
                    continue
                if metric == 'kits':
                    kit_threshold = value
                elif metric == 'anc':
                    anc_threshold = value
                elif metric == 'anc2':
                    anc2_threshold = value
                elif metric == 'high_risk':
                    high_risk_threshold = value
                else:
                    print("Unknown metric. Options: kits, anc, anc2, high_risk")
                    continue
                print("Threshold updated.")
            else:
                print("Invalid usage. set_threshold <metric> <value>")

        elif cmd == 'run_analysis':
            print("Running analysis...")
            insights = analyze_data(transformed_df, config[current_dataset_name])
            if insights:
                print(insights)

        elif cmd.startswith('find_anomalies '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metric = parts[1].strip()
                metrics_to_check = config[current_dataset_name].get('metrics_to_calculate', [])
                if metric in metrics_to_check:
                    anomalies, message = find_anomalies(transformed_df.copy(), metric)
                    print(message)
                    if anomalies is not None and not anomalies.empty:
                        dcol = district_col or resolve_column(transformed_df, configured_district, preferred_candidates=preferred)
                        if dcol:
                            print(anomalies[[dcol, metric, 'z_score']].to_string(index=False))
                        else:
                            print(anomalies.to_string(index=False))
                else:
                    print("Metric not in configured metrics_to_calculate.")
            else:
                print("Invalid usage. find_anomalies <metric_name>")

        elif cmd.startswith('predict '):
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
                    print("Invalid metric. Options: kits | high_risk")
            else:
                print("Invalid usage. predict <metric>")

        elif cmd.startswith('root_cause '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                guess = parts[1].strip()
                # fuzzy match metric name
                candidates = transformed_df.columns.tolist()
                match = difflib.get_close_matches(guess, candidates, n=1, cutoff=0.5)
                if not match:
                    print(f"Metric '{guess}' not found. Try 'list_metrics' to see available metrics.")
                else:
                    metric = match[0]
                    print(f"Running root cause analysis for '{metric}' (matched from '{guess}')...")
                    try:
                        correlations = run_root_cause_analysis(transformed_df.copy(), metric)
                        print(correlations)
                    except Exception as e:
                        print("Error running root cause analysis:", e)
            else:
                print("Invalid usage. root_cause <metric_name>")

        elif cmd.startswith('generate_dashboard '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                metrics = [m.strip() for m in parts[1].split(',')]
                try:
                    create_dashboard(transformed_df.copy(), metrics)
                except Exception as e:
                    print("Error creating dashboard:", e)
            else:
                print("Invalid usage. generate_dashboard <metric1,metric2,...>")

        elif cmd.startswith('dashboard_for '):
            parts = command.split(' ', 2)
            if len(parts) == 3:
                district_name = parts[1].strip().title()
                metrics = [m.strip() for m in parts[2].split(',')]
                output_filename = f"data/{district_name.replace(' ', '_')}_dashboard.png"
                try:
                    create_district_dashboard(transformed_df.copy(), district_name, metrics, output_filename)
                except Exception as e:
                    print("Error creating district dashboard:", e)
            else:
                print("Invalid usage. dashboard_for <district_name> <metric1,metric2,...>")

        elif cmd.startswith('set_dataset '):
            parts = command.split(' ', 1)
            if len(parts) > 1:
                ds = parts[1].strip()
                if ds in config:
                    # update current dataset and run pipeline for it
                    globals()['current_dataset_name'] = ds
                    print(f"Switching to dataset '{ds}'. Running pipeline...")
                    cleaned, transformed = run_pipeline(ds)
                    if transformed is not None:
                        # update TRANSFORMED_DATA_PATH fallback and reload transformed_df
                        try:
                            transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)
                        except Exception:
                            pass
                        # set function-level transformed_df to new dataframe
                        try:
                            # set the module-global transformed_df so CLI uses new data
                            globals()['transformed_df'] = transformed
                        except Exception:
                            pass
                        print(f"Dataset switched to '{ds}'. You can now run commands against it.")
                    else:
                        print("Pipeline failed for new dataset. Current dataset remains unchanged.")
                else:
                    print(f"Dataset '{ds}' not found in config.")
            else:
                print("Invalid usage. set_dataset <dataset_name>")

        elif cmd == 'help':
            print("\nAvailable commands:")
            print("  list_metrics                            - List available metrics/columns in the loaded dataset")
            print("  get_insights <district_name> <metric_name>")
            print("  set_threshold <metric> <value>")
            print("  run_analysis")
            print("  find_anomalies <metric>")
            print("  predict <metric>")
            print("  generate_dashboard <metric1,metric2,...>")
            print("  dashboard_for <district_name> <metric1,metric2,...>")
            print("  root_cause <metric_name>")
            print("  set_dataset <dataset_name>")
            print("  exit / quit")
        else:
            print("Unknown command. Type 'help' for usage.")

# -----------------------
# Runner that loads config and starts pipeline + CLI
# -----------------------
def run_pipeline_and_start_cli():
    global config, current_dataset_name, transformed_df, cleaned_df_global

    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Error: config.yaml not found. Aborting.")
        return

    print("[Policymaker CLI] --> [RTGS Agent]")
    print(f"Using dataset: {current_dataset_name}")

    # run pipeline for current dataset
    cleaned, transformed = run_pipeline(current_dataset_name, interactive=True)
    if transformed is None:
        print("Pipeline failed to produce transformed data. Aborting CLI startup.")
        return

    # save state to global csv path for interactive mode fallback
    try:
        transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)
    except Exception:
        pass

    print("\nPipeline complete. Starting interactive CLI.")
    run_interactive_mode()

if __name__ == '__main__':
    run_pipeline_and_start_cli()
