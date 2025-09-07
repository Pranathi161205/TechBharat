# main.py — Clean, unified rewrite
# --------------------------------
# Features:
# - Single import block & globals
# - Robust column resolution utilities
# - Pipeline runner for any dataset in config.yaml
# - Executive summary generators for multiple dataset types
# - Interactive CLI with explicit commands (and optional NLP)
# - Safe fallbacks when config keys/columns are missing

import os
import sys
import difflib
import pandas as pd
import yaml
from typing import Optional, List, Dict, Any

from scripts.clean_data import clean_data, load_data
from scripts.transform_data import transform_data
from scripts.analyze_data import (
    analyze_data,
    find_anomalies,
    generate_executive_summary,
    run_root_cause_analysis,
)
from scripts.predict_data import predict_future_kits, predict_high_risk
from scripts.visualize_data import create_choropleth_map
from scripts.generate_report import generate_html_report
from scripts.dashboard import create_dashboard, create_district_dashboard

# Optional: natural language command parser
try:
    from scripts.nlp_parser import parse_command  # returns dict with keys: command, district, metrics
except Exception:
    parse_command = None  # optional

# -----------------------
# Constants & Globals
# -----------------------
DATA_DIR = "data"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "data_cleaned.csv")
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, "data_transformed.csv")

config: Optional[Dict[str, Any]] = None
current_dataset_name = "health_data"

# Dataframes
transformed_df: Optional[pd.DataFrame] = None
cleaned_df_global: Optional[pd.DataFrame] = None

# Thresholds (editable in CLI)
kit_threshold = 0.8
anc_threshold = 0.5
anc2_threshold = 0.9
high_risk_threshold = 0.1

# -----------------------
# Utilities
# -----------------------
def _normalize_for_lookup(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\u00b0", "deg").replace("°", "deg").replace("Â", "")
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch == "_")
    s = "_".join(s.split())
    return s


def pick_column(df: pd.DataFrame, logical_name: str, preferred: Optional[List[str]] = None) -> Optional[str]:
    """Resolve a column using preferred names, normalized matches, substring/token overlap."""
    if df is None or df.columns is None:
        return None
    cols = list(df.columns)
    if preferred:
        for cand in preferred:
            if cand in cols:
                return cand
    ln = _normalize_for_lookup(logical_name)
    for c in cols:
        if _normalize_for_lookup(c) == ln:
            return c
    for c in cols:
        cn = _normalize_for_lookup(c)
        if ln in cn or cn in ln:
            return c
    # token overlap
    t_tokens = set(ln.split("_"))
    best, best_score = None, 0
    for c in cols:
        score = len(t_tokens.intersection(set(_normalize_for_lookup(c).split("_"))))
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None


# -----------------------
# Executive summary builders
# -----------------------
def generate_and_save_summary(dataset_name: str, df_transformed: pd.DataFrame) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{dataset_name}_executive_summary.txt")
    try:
        if dataset_name == "health_data":
            try:
                res = generate_executive_summary(df_transformed, None)
                summary_text = res if isinstance(res, str) else _build_health_summary(df_transformed)
            except Exception:
                summary_text = _build_health_summary(df_transformed)
        elif dataset_name == "temperature_data":
            summary_text = _build_temperature_summary(df_transformed)
        elif dataset_name == "tourism_domestic":
            summary_text = _build_tourism_summary(df_transformed, dataset_name)
        else:
            summary_text = _build_generic_summary(dataset_name, df_transformed)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        print(f"Executive summary saved to: {out_path}")
        return out_path
    except Exception as e:
        print("Could not create executive summary file:", e)
        return ""


def _build_health_summary(df: pd.DataFrame) -> str:
    parts = ["Report: Health dataset — executive summary."]
    dcol = pick_column(df, "district", ["district", "District", "district_name", "districtName"]) or "district"
    if "kit_coverage_ratio" in df.columns:
        top = df.sort_values("kit_coverage_ratio", ascending=False).head(3)
        tops = [f"{row.get(dcol, getattr(row, dcol, 'Unknown'))} ({row['kit_coverage_ratio']:.2f})" for _, row in top.iterrows()]
        parts.append(f"Top kit coverage districts: {', '.join(tops)}.")
    if "high_risk_ratio" in df.columns:
        highest = df.sort_values("high_risk_ratio", ascending=False).head(3)
        highs = [f"{row.get(dcol, getattr(row, dcol, 'Unknown'))} ({row['high_risk_ratio']:.2f})" for _, row in highest.iterrows()]
        parts.append(f"Highest high-risk ratios: {', '.join(highs)}.")
    parts.append("Recommendation: prioritize outreach and kit distribution in high-risk and low-coverage districts; validate data completeness where ratios are unexpectedly low or high.")
    return " ".join(parts)


def _build_temperature_summary(df: pd.DataFrame) -> str:
    parts = ["Report: Temperature dataset — executive summary."]
    dcol = pick_column(df, "district", ["district", "District", "district_name", "districtName"]) or "district"
    avg_col = pick_column(df, "avg_temp", ["avg_temp", "average_temp", "avg temperature"])
    max_col = pick_column(df, "max_temperature", ["max_temperature", "max temp", "maximum_temperature"])
    min_col = pick_column(df, "min_temperature", ["min_temperature", "min temp", "minimum_temperature"])
    if avg_col and avg_col in df.columns and pd.api.types.is_numeric_dtype(df[avg_col]):
        parts.append(f"Overall average temperature (district averages): {df[avg_col].mean():.2f}°.")
    if max_col and max_col in df.columns:
        top_hot = df.sort_values(max_col, ascending=False).head(3)
        hot = [f"{row.get(dcol, getattr(row, dcol, 'Unknown'))} ({row[max_col]})" for _, row in top_hot.iterrows()]
        parts.append(f"Top hottest districts: {', '.join(hot)}.")
    if min_col and min_col in df.columns:
        top_cold = df.sort_values(min_col, ascending=True).head(3)
        cold = [f"{row.get(dcol, getattr(row, dcol, 'Unknown'))} ({row[min_col]})" for _, row in top_cold.iterrows()]
        parts.append(f"Top coldest districts: {', '.join(cold)}.")
    parts.append("Recommendation: review extreme-temperature districts for mitigation and cross-check sensor quality if values look anomalous.")
    return " ".join(parts)


def _build_tourism_summary(df: pd.DataFrame, dataset_name: str) -> str:
    parts = [f"Report: Tourism Domestic Visitors — dataset: {dataset_name}."]
    dcol = pick_column(df, "district", ["district", "District", "district_name", "districtName"]) or "district"
    total_col = pick_column(df, "total_visitors", ["total_visitors", "Total Visitors", "visitors"])
    if total_col and total_col in df.columns:
        total = int(pd.to_numeric(df[total_col], errors="coerce").fillna(0).sum())
        parts.append(f"In 2024, the dataset records a combined total of {total:,} domestic visitors across reported districts.")
        top3 = df.sort_values(total_col, ascending=False).head(3)
        t3 = [f"{row.get(dcol, getattr(row, dcol, 'Unknown'))} ({int(row[total_col]):,})" for _, row in top3.iterrows()]
        parts.append(f"The top districts by visitor volume are: {', '.join(t3)}.")
    peak_col = pick_column(df, "peak_month", ["peak_month", "Peak Month", "month"])
    if peak_col and peak_col in df.columns:
        pm = df[peak_col].dropna()
        if not pm.empty:
            mode = pm.mode()
            if not mode.empty:
                parts.append(f"Peak season signal: many districts show {mode.iloc[0]} as their highest-visitor month.")
    parts.append("Recommendation: consider targeted capacity and marketing actions for the top districts during peak months; investigate districts with unexpectedly low reporting counts to improve data completeness.")
    return " ".join(parts)


def _build_generic_summary(dataset_name: str, df: pd.DataFrame) -> str:
    parts = [f"Report: {dataset_name} — executive summary.", f"Rows: {len(df)}; Columns: {len(df.columns)}."]
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric:
        parts.append(f"Key numeric column (sample): {numeric[0]}.")
    parts.append("Recommendation: inspect transformed dataset for domain-specific KPIs to populate a richer executive summary.")
    return " ".join(parts)


# -----------------------
# Pipeline runner
# -----------------------
def run_pipeline(dataset_name: str, interactive: bool = False):
    global transformed_df, cleaned_df_global, config, current_dataset_name

    if config is None:
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                globals()["config"] = config
        except FileNotFoundError:
            print("Error: config.yaml not found. Aborting pipeline.")
            return None, None

    if dataset_name not in config:
        print(f"Error: dataset '{dataset_name}' not found in config.")
        return None, None

    dataset_config = config[dataset_name]
    file_path = dataset_config.get("file_path")
    if not file_path:
        print(f"Error: 'file_path' not defined for dataset '{dataset_name}' in config.yaml.")
        return None, None

    print(f"[Pipeline] 1. Loading {dataset_name} data...")
    raw_df = load_data(file_path)
    if raw_df is None:
        return None, None

    print(f"[Pipeline] 2. Cleaning & Standardizing {dataset_name} data...")
    cleaned_df = clean_data(raw_df, dataset_config.get("columns", {}))
    if cleaned_df is None:
        return None, None

    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned_out = os.path.join(DATA_DIR, f"{dataset_name}_cleaned.csv")
    try:
        cleaned_df.to_csv(cleaned_out, index=False)
    except Exception:
        pass
    cleaned_df_global = cleaned_df

    print(f"[Pipeline] 3. Transforming {dataset_name} data...")
    transformed = transform_data(cleaned_df, dataset_config, dataset_name)
    if transformed is None:
        return cleaned_df, None

    transformed_out = os.path.join(DATA_DIR, f"{dataset_name}_transformed.csv")
    try:
        transformed.to_csv(transformed_out, index=False)
    except Exception:
        pass
    transformed_df = transformed

    print(f"[Pipeline] 4. Analyzing data & generating initial insights...")
    try:
        insights = analyze_data(transformed, dataset_config)
        if insights:
            print("\nInitial Insights:")
            print(insights)
    except Exception as e:
        print(f"   - Error during analysis: {e}")

    # tourism-specific friendly prints
    if dataset_name == "tourism_domestic":
        try:
            tdf = transformed.copy()
            tv = pick_column(tdf, "total_visitors", ["total_visitors", "Total Visitors"]) or "total_visitors"
            if tv in tdf.columns:
                top5 = tdf.sort_values(tv, ascending=False).head(5)
                print("\nTop 5 districts by total visitors (2024):")
                dcol = pick_column(tdf, "district", ["district", "District", "district_name"]) or "district"
                for i, row in enumerate(top5[[dcol, tv]].to_dict(orient="records"), start=1):
                    print(f"  {i}. {row[dcol]}: {int(row[tv]):,}")
            # save insights csv
            insights_out = os.path.join(DATA_DIR, f"{dataset_name}_insights.csv")
            if tv in tdf.columns:
                tdf.sort_values(tv, ascending=False).to_csv(insights_out, index=False)
                print(f"\nTourism insights saved to: {insights_out}")
            # try dashboard
            try:
                create_dashboard(tdf.copy(), [c for c in [tv, pick_column(tdf, "avg_monthly_visitors")] if c and c in tdf.columns])
            except Exception:
                # fallback small bar chart
                try:
                    import matplotlib.pyplot as plt
                    out_img = os.path.join(DATA_DIR, f"{dataset_name}_top20.png")
                    plot_df = tdf.sort_values(tv, ascending=False).head(20) if tv in tdf.columns else tdf.head(20)
                    if tv in plot_df.columns:
                        plt.figure(figsize=(10, 8))
                        plt.barh(plot_df[pick_column(plot_df, "district", ["district", "District"])], plot_df[tv])
                        plt.gca().invert_yaxis()
                        plt.xlabel("Total Visitors (2024)")
                        plt.title("Top 20 districts by domestic visitors (2024)")
                        plt.tight_layout()
                        plt.savefig(out_img, dpi=150)
                        plt.close()
                        print(f"Tourism dashboard image saved to: {out_img}")
                except Exception:
                    pass
        except Exception as e:
            print("Error creating tourism-specific insights:", e)

    print("\n[Pipeline] 5. Running predictive analysis (if available)...")
    try:
        if dataset_name == "health_data":
            predicted_kits, pred_date_kits = predict_future_kits(cleaned_df.copy())
            predicted_high_risk_val, pred_date_hr = predict_high_risk(cleaned_df.copy())
            print(f"   - Predicted MCH kits for {pred_date_kits}: {predicted_kits}")
            print(f"   - Predicted high-risk pregnancies for {pred_date_hr}: {predicted_high_risk_val}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")

    print("\n[Pipeline] 6. Creating geospatial visualization (if available)...")
    try:
        if dataset_name == "health_data":
            create_choropleth_map(transformed)
        else:
            print("   - Geospatial visualization not configured for this dataset.")
    except Exception as e:
        print(f"   - Error during visualization: {e}")

    print("\n[Pipeline] 7. Creating dashboard visualization (if available)...")
    try:
        if dataset_name == "health_data":
            default_metrics = [m for m in ["kit_coverage_ratio", "high_risk_ratio"] if m in transformed.columns]
            if default_metrics:
                create_dashboard(transformed.copy(), default_metrics)
    except Exception as e:
        print(f"   - Error during dashboard creation: {e}")

    # Generate and save executive summary
    try:
        generate_and_save_summary(dataset_name, transformed)
    except Exception as e:
        print("Error generating or saving executive summary:", e)

    # Optional: run alert check if alerting module exists (safe, non-fatal)
    try:
        scripts_dir = os.path.join(os.getcwd(), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import alerting  # type: ignore
        if hasattr(alerting, "check_and_send_alerts"):
            try:
                alerting.check_and_send_alerts(dataset_name, transformed, config=config)
            except Exception as ee:
                print("Alerting check (optional) failed:", ee)
    except Exception:
        # alerting not present; ignore
        pass

    print("\n[Pipeline] Completed for dataset:", dataset_name)
    return cleaned_df, transformed


# -----------------------
# Interactive CLI
# -----------------------
def run_interactive_mode():
    global transformed_df, cleaned_df_global, config, current_dataset_name

    if transformed_df is None:
        try:
            transformed_df_local = pd.read_csv(TRANSFORMED_DATA_PATH)
            transformed_df = transformed_df_local  # type: ignore
            print("Interactive mode started. Transformed data loaded.")
        except FileNotFoundError:
            print("Error: Transformed data not found. Please run the pipeline first.")
            return

    def read_summary_and_print(ds_name: str):
        path = os.path.join(DATA_DIR, f"{ds_name}_executive_summary.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print("\n" + f.read())
        else:
            print(f"No summary for {ds_name}. Run `set_dataset {ds_name}` to generate it.")

    print("\nRTGS-CLI started. Type 'help' for available commands. Type 'exit' to quit.")
    while True:
        user_input = input("\nRTGS-CLI> ").strip()
        if not user_input:
            continue
        cmd = user_input.strip()

        if cmd.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if cmd.lower() == "help":
            print(
                "Commands:\n"
                "  list_metrics\n"
                "  show_summary\n"
                "  set_dataset <name>\n"
                "  get_insights <district> <metric>\n"
                "  run_analysis\n"
                "  find_anomalies <metric>\n"
                "  predict <kits|high_risk>\n"
                "  root_cause <metric>\n"
                "  generate_dashboard <m1,m2,...>\n"
                "  dashboard_for <district> <m1,m2,...>\n"
                "  exit\n"
            )
            continue

        # simple explicit commands
        if cmd.lower() == "list_metrics":
            print("Available metrics / columns:")
            print(", ".join([str(c) for c in transformed_df.columns.tolist()]))
            continue

        if cmd.lower() == "show_summary":
            read_summary_and_print(current_dataset_name)
            continue

        if cmd.lower().startswith("set_dataset"):
            parts = cmd.split(" ", 1)
            if len(parts) == 2:
                ds = parts[1].strip()
                if ds in config:
                    current_dataset_name = ds
                    print(f"Switching to dataset '{ds}'. Running pipeline...")
                    cleaned, transformed = run_pipeline(ds)
                    if transformed is not None:
                        try:
                            transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)
                        except Exception:
                            pass
                        globals()["transformed_df"] = transformed
                        print(f"Dataset switched to '{ds}'.")
                        read_summary_and_print(ds)
                    else:
                        print("Pipeline failed for new dataset.")
                else:
                    print(f"Dataset '{ds}' not found in config.yaml.")
            else:
                print("Usage: set_dataset <dataset_name>")
            continue

        if cmd.lower().startswith("get_insights "):
            parts = cmd.split(" ", 2)
            if len(parts) == 3:
                district = parts[1].strip().title()
                metric = parts[2].strip()
                # fuzzy-match metric
                if metric not in transformed_df.columns:
                    match = difflib.get_close_matches(metric, transformed_df.columns.tolist(), n=1, cutoff=0.6)
                    if match:
                        metric = match[0]
                        print(f"Using metric '{metric}' (fuzzy matched).")
                    else:
                        print(f"Metric '{metric}' not found.")
                        continue
                dcol = pick_column(transformed_df, "district", ["district", "districtName", "district_name"]) or "district"
                res = transformed_df[transformed_df[dcol].astype(str).str.strip().str.title() == district]
                if not res.empty:
                    print(res[[dcol, metric]].to_string(index=False))
                else:
                    print(f"No data for district '{district}'.")
            else:
                print("Usage: get_insights <district> <metric>")
            continue

        if cmd.lower() == "run_analysis":
            print("Running analysis...")
            try:
                insights = analyze_data(transformed_df, config.get(current_dataset_name, {}))
                if insights:
                    print(insights)
            except Exception as e:
                print("Analysis error:", e)
            continue

        if cmd.lower().startswith("find_anomalies "):
            _, metric = cmd.split(" ", 1)
            try:
                anomalies, message = find_anomalies(transformed_df.copy(), metric)
                print(message)
                if anomalies is not None and not anomalies.empty:
                    print(anomalies.to_string(index=False))
            except Exception as e:
                print("Anomaly detection error:", e)
            continue

        if cmd.lower().startswith("predict "):
            _, metric = cmd.split(" ", 1)
            try:
                if metric.lower() == "kits":
                    pred, date = predict_future_kits(cleaned_df_global.copy())
                    print(f"Predicted MCH kits for {date}: {pred}")
                elif metric.lower() == "high_risk":
                    pred, date = predict_high_risk(cleaned_df_global.copy())
                    print(f"Predicted high-risk pregnancies for {date}: {pred}")
                else:
                    print("Options: kits | high_risk")
            except Exception as e:
                print("Prediction error:", e)
            continue

        if cmd.lower().startswith("root_cause "):
            _, guess = cmd.split(" ", 1)
            match = difflib.get_close_matches(guess, transformed_df.columns.tolist(), n=1, cutoff=0.5)
            if not match:
                print(f"Metric '{guess}' not found. Use list_metrics to inspect available columns.")
                continue
            metric = match[0]
            print(f"Running root cause analysis for '{metric}'...")
            try:
                corr = run_root_cause_analysis(transformed_df.copy(), metric)
                print(corr)
            except Exception as e:
                print("Root cause error:", e)
            continue

        if cmd.lower().startswith("generate_dashboard "):
            _, rest = cmd.split(" ", 1)
            metrics = [m.strip() for m in rest.split(",") if m.strip()]
            try:
                create_dashboard(transformed_df.copy(), metrics)
                print("Dashboard generated (if supported by dashboard module).")
            except Exception as e:
                print("Dashboard error:", e)
            continue

        if cmd.lower().startswith("dashboard_for "):
            parts = cmd.split(" ", 2)
            if len(parts) == 3:
                district = parts[1].strip().title()
                metrics = [m.strip() for m in parts[2].split(",") if m.strip()]
                out = f"data/{district.replace(' ', '_')}_dashboard.png"
                try:
                    create_district_dashboard(transformed_df.copy(), district, metrics, out)
                    print(f"District dashboard saved to: {out}")
                except Exception as e:
                    print("District dashboard error:", e)
            else:
                print("Usage: dashboard_for <district> <metric1,metric2,...>")
            continue

        # fallback: try NLP parser if available
        if parse_command is not None:
            try:
                parsed = parse_command(user_input, config.get(current_dataset_name, {}).get("metrics_to_calculate", []))
                if isinstance(parsed, dict) and parsed.get("command"):
                    cmdname = parsed["command"]
                    if cmdname == "get_insights" and parsed.get("district") and parsed.get("metrics"):
                        district = parsed["district"]
                        metric = parsed["metrics"][0]
                        # reuse logic above
                        user_like = f"get_insights {district} {metric}"
                        # simple recursive call imitation: set cmd and let loop handle - here call directly:
                        parts = user_like.split(" ", 2)
                        district = parts[1].strip().title()
                        metric = parts[2].strip()
                        if metric not in transformed_df.columns:
                            match = difflib.get_close_matches(metric, transformed_df.columns.tolist(), n=1, cutoff=0.6)
                            if match:
                                metric = match[0]
                        dcol = pick_column(transformed_df, "district", ["district", "districtName", "district_name"]) or "district"
                        res = transformed_df[transformed_df[dcol].astype(str).str.strip().str.title() == district]
                        if not res.empty:
                            print(res[[dcol, metric]].to_string(index=False))
                        else:
                            print(f"No data for district '{district}'.")
                        continue
            except Exception:
                pass

        print("Unknown command. Type 'help' for usage.")


# -----------------------
# Runner
# -----------------------
def run_pipeline_and_start_cli():
    global config, current_dataset_name, transformed_df, cleaned_df_global

    try:
        with open("config.yaml", "r") as f:
            c = yaml.safe_load(f)
            globals()["config"] = c
            config = c
            print("Configuration loaded successfully.")
    except FileNotFoundError:
        print("Error: config.yaml not found. Aborting.")
        return

    print("[Policymaker CLI] --> [RTGS Agent]")
    print(f"Using dataset: {current_dataset_name}")

    cleaned, transformed = run_pipeline(current_dataset_name, interactive=True)
    if transformed is None:
        print("Pipeline failed to produce transformed data. Aborting CLI startup.")
        return

    try:
        transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)
    except Exception:
        pass

    # ensure globals set
    globals()["transformed_df"] = transformed
    globals()["cleaned_df_global"] = cleaned

    print("\nPipeline complete. Starting interactive CLI.")
    run_interactive_mode()


if __name__ == "__main__":
    run_pipeline_and_start_cli()

#!/usr/bin/env python3
"""
main.py - controller for rtgs-cli.py with simple dataset analysis support.

Features added:
 - --analysis run_analysis  : runs dataset-specific analysis (supports tourism_domestic)
 - --report-text "<text>"  : optionally provide raw report text to parse instead of querying CLI
"""
import argparse
import subprocess
import threading
import sys
import re
import time
from queue import Queue, Empty

RTGS_CLI_CMD = ["python", "-u", "rtgs-cli.py"]  # adjust if needed

def enqueue_output(pipe, queue):
    for line in iter(pipe.readline, b''):
        queue.put(line.decode(errors='replace'))
    pipe.close()

def run_commands_and_capture(cmds, timeout=None):
    proc = subprocess.Popen(RTGS_CLI_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    q = Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q), daemon=True)
    t.start()

    out_lines = []
    try:
        for c in cmds:
            proc.stdin.write((c + "\n").encode())
            proc.stdin.flush()
            time.sleep(0.25)

        end_time = time.time() + (timeout or 3.0)
        while time.time() < end_time or not q.empty():
            try:
                line = q.get(timeout=0.1)
                out_lines.append(line)
                print(line, end="")  # stream to console
            except Empty:
                pass
    finally:
        try:
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()
        except Exception:
            pass
        proc.terminate()
        proc.wait()

    return "".join(out_lines)

# ---------------------------
# Analysis helpers
# ---------------------------

def parse_tourism_domestic_report(text):
    """
    Parse report like:
    'In 2024, the dataset records a combined total of 88,239,675 domestic visitors ... The top districts by visitor volume are: Hyderabad,Ranga Reddy,Medchal - Malkajigiri,Vikarabad (21,739,960), Rajanna Sircilla (16,288,424), Mulugu (14,610,348).'
    Returns dict with total and list of (district, visitors).
    """
    res = {"dataset": "tourism_domestic", "year": None, "total_visitors": None, "top_districts": []}

    # year and total
    m_total = re.search(r"In\s+(\d{4}).*combined total of\s*([\d,]+)\s*domestic visitors", text, re.IGNORECASE)
    if m_total:
        res["year"] = int(m_total.group(1))
        res["total_visitors"] = int(m_total.group(2).replace(",", ""))

    # top districts block (attempt to find district (number) pairs)
    # allow patterns like "DistrictA (21,739,960), DistrictB (16,288,424), DistrictC (14,610,348)"
    pairs = re.findall(r"([A-Za-z0-9\-\s&\.]+?)\s*\(\s*([\d,]+)\s*\)", text)
    if pairs:
        for name, num in pairs:
            name = name.strip().rstrip(",")
            try:
                val = int(num.replace(",", ""))
            except:
                val = None
            res["top_districts"].append((name, val))

    # fallback if there is a CSV-like sequence before parentheses (e.g., "Hyderabad,Ranga Reddy,Medchal - Malkajigiri,Vikarabad (21,739,960)")
    if not res["top_districts"]:
        m_seq = re.search(r"top districts by visitor volume are:\s*([^\n\.]+)", text, re.IGNORECASE)
        if m_seq:
            seq = m_seq.group(1).strip()
            # find trailing numeric in parentheses and attribute to the preceding district
            # naive split by comma
            parts = [p.strip() for p in seq.split(",") if p.strip()]
            # try to attach numbers if present in the last parts
            for p in parts:
                mnum = re.search(r"(.+?)\s*\(\s*([\d,]+)\s*\)", p)
                if mnum:
                    n = mnum.group(1).strip()
                    v = int(mnum.group(2).replace(",", ""))
                    res["top_districts"].append((n, v))
                else:
                    # unknown number; put None
                    res["top_districts"].append((p, None))

    return res

def format_tourism_domestic_analysis(parsed):
    lines = []
    lines.append("=== Tourism Domestic Visitors — Analysis ===")
    lines.append(f"Dataset: {parsed.get('dataset')}")
    if parsed.get("year"):
        lines.append(f"Year: {parsed.get('year')}")
    if parsed.get("total_visitors") is not None:
        lines.append(f"Combined total visitors (reported districts): {parsed['total_visitors']:,}")
    lines.append("")
    if parsed.get("top_districts"):
        lines.append("Top districts by visitor volume (reported):")
        for i, (name, val) in enumerate(parsed["top_districts"], start=1):
            if val:
                lines.append(f"  {i}. {name} — {val:,}")
            else:
                lines.append(f"  {i}. {name} — (value not found)")
    lines.append("")
    # Recommendations (basic templates)
    lines.append("Recommendations:")
    lines.append("- Consider targeted capacity and marketing actions for the top districts during peak months.")
    lines.append("- Investigate districts with unexpectedly low reporting counts to improve data completeness and quality.")
    lines.append("- If tourist spikes correlate with services, plan seasonal staffing and resource allocation accordingly.")
    return "\n".join(lines)

# ---------------------------
# Dataset-specific analysis runner
# ---------------------------

def run_analysis_for_dataset(dataset, report_text=None):
    dataset = dataset.lower()
    if dataset == "tourism_domestic":
        if report_text:
            parsed = parse_tourism_domestic_report(report_text)
            print(format_tourism_domestic_analysis(parsed))
            return

        # Otherwise, query rtgs-cli to compute numbers
        # The following commands are examples; adapt to rtgs-cli available commands.
        cmds = [
            f"set_dataset {dataset}",
            # ask CLI to summarize totals and top districts; adapt the get_insights command to your CLI syntax
            "get_insights total date visitors --year 2024",
            "get_insights top district visitors --top 10 --year 2024"
        ]
        print("Running CLI commands to compute tourism_domestic analysis...")
        raw = run_commands_and_capture(cmds, timeout=5.0)
        # Try to parse the CLI output for totals/top districts using the same parser as above
        parsed = parse_tourism_domestic_report(raw)
        # If parser failed, also try to extract patterns like "Total: 88,239,675" or "Top 1: Hyderabad (21,739,960)"
        if parsed.get("total_visitors") is None:
            m_total = re.search(r"Total[^\d]*([\d,]{6,})", raw)
            if m_total:
                parsed["total_visitors"] = int(m_total.group(1).replace(",", ""))

        # simple extraction for "Top" lines
        tops = re.findall(r"Top\s*\d+\s*[:\-]\s*([A-Za-z0-9\-\s&\.]+?)\s*\(\s*([\d,]+)\s*\)", raw)
        if tops:
            parsed["top_districts"] = [(n.strip(), int(v.replace(",", ""))) for n, v in tops]

        print(format_tourism_domestic_analysis(parsed))
    else:
        print(f"No dataset-specific analysis defined for '{dataset}'. You can add it in run_analysis_for_dataset().")

# ---------------------------
# CLI wrapper reuse
# ---------------------------

def run_commands_and_capture(cmds, timeout=None):
    # minimal wrapper re-declared to keep single-file example; you can reuse the earlier one
    proc = subprocess.Popen(RTGS_CLI_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    q = Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q), daemon=True)
    t.start()

    out_lines = []
    try:
        for c in cmds:
            proc.stdin.write((c + "\n").encode())
            proc.stdin.flush()
            time.sleep(0.25)
        end_time = time.time() + (timeout or 3.0)
        while time.time() < end_time or not q.empty():
            try:
                line = q.get(timeout=0.1)
                out_lines.append(line)
                print(line, end="")
            except Empty:
                pass
    finally:
        try:
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()
        except Exception:
            pass
        proc.terminate()
        proc.wait()
    return "".join(out_lines)

# ---------------------------
# CLI arg parsing
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="main.py controller with analysis support")
    parser.add_argument("--analysis", type=str, help="Run named analysis (e.g., run_analysis)")
    parser.add_argument("--dataset", type=str, help="Dataset name for analysis (e.g., tourism_domestic)")
    parser.add_argument("--report-text", type=str, help="Optional raw report text to parse instead of querying CLI")
    args = parser.parse_args()

    if args.analysis == "run_analysis":
        if not args.dataset:
            print("Please provide --dataset <name> for run_analysis.")
            return
        run_analysis_for_dataset(args.dataset, report_text=args.report_text)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
