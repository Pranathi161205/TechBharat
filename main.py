#!/usr/bin/env python3
"""
main.py - unified controller and interactive RTGS agent

Features:
- Single import block & globals
- Robust column resolution utilities
- Pipeline runner for any dataset in config.yaml (fallback DEFAULT_CONFIG provided)
- Executive summary generators for multiple dataset types
- Interactive CLI with explicit commands (and optional NLP)
- Safe fallbacks when config keys/columns are missing
- Tourism-specific report parser and CLI-runner utilities
"""

import os
import sys
import time
import re
import difflib
import argparse
import threading
import subprocess
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import yaml

# Domain-specific modules (expected under ./scripts). Wrap optional features in try/except where appropriate.
try:
    from scripts.clean_data import clean_data, load_data
except Exception:
    def load_data(path):
        raise RuntimeError("scripts.clean_data.load_data not available")

    def clean_data(df, cols):
        raise RuntimeError("scripts.clean_data.clean_data not available")

try:
    from scripts.transform_data import transform_data
except Exception:
    def transform_data(df, config, name=None):
        # identity transform fallback
        return df.copy()

try:
    from scripts.analyze_data import (
        analyze_data,
        find_anomalies,
        generate_executive_summary,
        run_root_cause_analysis,
    )
except Exception:
    def analyze_data(df, cfg):
        return None

    def find_anomalies(df, metric):
        return None, "Anomaly detection not available"

    def generate_executive_summary(df, cfg):
        return None

    def run_root_cause_analysis(df, metric):
        return {"error": "root cause module missing"}

try:
    from scripts.predict_data import predict_future_kits, predict_high_risk
except Exception:
    def predict_future_kits(df):
        return None, None

    def predict_high_risk(df):
        return None, None

try:
    from scripts.visualize_data import create_choropleth_map
except Exception:
    def create_choropleth_map(df):
        raise RuntimeError("visualization module not available")

try:
    from scripts.generate_report import generate_html_report
except Exception:
    def generate_html_report(*args, **kwargs):
        raise RuntimeError("report generator not available")

try:
    from scripts.dashboard import create_dashboard, create_district_dashboard
except Exception:
    def create_dashboard(*args, **kwargs):
        raise RuntimeError("dashboard module not available")

    def create_district_dashboard(*args, **kwargs):
        raise RuntimeError("district dashboard not available")

# Optional: natural language command parser (best-effort import)
try:
    from scripts.nlp_parser import parse_command  # returns dict with keys: command, district, metrics
except Exception:
    parse_command = None  # optional and non-fatal

# -----------------------
# Constants & Globals
# -----------------------
DATA_DIR = "data"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "data_cleaned.csv")
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, "data_transformed.csv")
CONFIG_PATH = "config.yaml"

RTGS_CLI_CMD = ["python", "-u", "rtgs-cli.py"]  # used if we call the CLI subprocess

config: Optional[Dict[str, Any]] = None
current_dataset_name: str = "health_data"

# Dataframes (globals used by CLI)
transformed_df: Optional[pd.DataFrame] = None
cleaned_df_global: Optional[pd.DataFrame] = None

# Thresholds (editable in CLI)
kit_threshold = 0.8
anc_threshold = 0.5
anc2_threshold = 0.9
high_risk_threshold = 0.1

# -----------------------
# DEFAULT_CONFIG + YAML helper
# -----------------------
DEFAULT_CONFIG = {
    "consumption_details": {
        "file_path": "data/consumption_details.csv",
        "columns": {
            "circle": "circle",
            "division": "division",
            "subdivision": "subdivision",
            "section": "section",
            "area": "area",
            "catdesc": "catdesc",
            "catcode": "catcode",
            "totservices": "totservices",
            "billdservices": "billdservices",
            "units": "units",
            "load": "load",
        },
        "display_name": "RTGS - Consumption Details",
        "metrics_to_calculate": ["totservices", "billdservices", "load", "units"],
    }
}


def safe_read_yaml(path: str) -> Optional[Dict[str, Any]]:
    """
    Safe YAML loader. Returns dict on success, None if file not found,
    or empty dict on parse success with no content.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: could not read YAML '{path}': {e}")
        return {}


# -----------------------
# Utility functions
# -----------------------
def _normalize_for_lookup(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("\u00b0", "deg").replace("°", "deg").replace("Â", "")
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace() or ch in ("_", "-"))
    s = "_".join(s.split())
    return s


def pick_column(df: pd.DataFrame, logical_name: str, preferred: Optional[List[str]] = None) -> Optional[str]:
    """
    Resolve a column using preferred names, exact normalized match, substring/token overlap.
    Returns the matching column name or None.
    """
    if df is None or df.columns is None:
        return None
    cols = list(df.columns)
    # preferred exact match in raw names
    if preferred:
        for cand in preferred:
            if cand in cols:
                return cand
    # exact normalized match
    ln = _normalize_for_lookup(logical_name)
    for c in cols:
        if _normalize_for_lookup(c) == ln:
            return c
    # substring matches
    for c in cols:
        cn = _normalize_for_lookup(c)
        if ln in cn or cn in ln:
            return c
    # token overlap scoring
    t_tokens = set(token for token in ln.split("_") if token)
    best, best_score = None, 0
    for c in cols:
        c_tokens = set(token for token in _normalize_for_lookup(c).split("_") if token)
        score = len(t_tokens.intersection(c_tokens))
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------
# Executive summary builders
# -----------------------
def generate_and_save_summary(dataset_name: str, df_transformed: pd.DataFrame) -> str:
    ensure_data_dir()
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
        elif dataset_name == "consumption_details":
            summary_text = _build_consumption_summary(df_transformed)
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
        tops = []
        for _, row in top.iterrows():
            name = row.get(dcol) if dcol in row else getattr(row, dcol, "Unknown")
            tops.append(f"{name} ({row['kit_coverage_ratio']:.2f})")
        parts.append(f"Top kit coverage districts: {', '.join(tops)}.")
    if "high_risk_ratio" in df.columns:
        highest = df.sort_values("high_risk_ratio", ascending=False).head(3)
        highs = []
        for _, row in highest.iterrows():
            name = row.get(dcol) if dcol in row else getattr(row, dcol, "Unknown")
            highs.append(f"{name} ({row['high_risk_ratio']:.2f})")
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
        t3 = []
        for _, row in top3.iterrows():
            name = row.get(dcol) if dcol in row else getattr(row, dcol, "Unknown")
            t3.append(f"{name} ({int(row[total_col]):,})")
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


def _build_consumption_summary(df: pd.DataFrame) -> str:
    parts = ["Report: Consumption Details — executive summary."]
    dcol = pick_column(df, "division", ["division", "Division", "circle", "circle_name"]) or "division"
    # total services
    tot_col = pick_column(df, "totservices", ["totservices", "total_services", "tot_services", "total"])
    if tot_col and tot_col in df.columns:
        total = int(pd.to_numeric(df[tot_col], errors="coerce").fillna(0).sum())
        parts.append(f"Combined total services (reported rows): {total:,}.")
        # top categories by billed services
        bill_col = pick_column(df, "billdservices", ["billdservices", "billed_services", "billd_services", "billed"])
        cat_col = pick_column(df, "catdesc", ["catdesc", "category", "catdesc", "catcode"])
        if bill_col and bill_col in df.columns and cat_col and cat_col in df.columns:
            agg = df.groupby(cat_col)[bill_col].sum().sort_values(ascending=False).head(5)
            tops = [f"{idx} ({int(val):,})" for idx, val in agg.items()]
            parts.append(f"Top categories by billed services: {', '.join(tops)}.")
    load_col = pick_column(df, "load", ["load", "utilization", "load_factor"])
    if load_col and load_col in df.columns and pd.api.types.is_numeric_dtype(df[load_col]):
        high_load = df.sort_values(load_col, ascending=False).head(3)
        hl = []
        for _, row in high_load.iterrows():
            name = row.get(dcol) if dcol in row else getattr(row, dcol, "Unknown")
            hl.append(f"{name} ({row[load_col]:.2f})")
        parts.append(f"Highest load areas: {', '.join(hl)}.")
    parts.append("Recommendation: investigate high-load areas for capacity planning; review category-level billing patterns for optimization; improve completeness where fields are missing.")
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
def run_pipeline(dataset_name: str, interactive: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    global transformed_df, cleaned_df_global, config, current_dataset_name

    # Load config if missing
    if config is None:
        c = safe_read_yaml(CONFIG_PATH)
        if c is None:
            print("Warning: config.yaml not found — using built-in DEFAULT_CONFIG for quick runs.")
            c = DEFAULT_CONFIG.copy()
        else:
            merged = DEFAULT_CONFIG.copy()
            merged.update(c)
            c = merged
        config = c
        globals()["config"] = config

    if dataset_name not in config:
        print(f"Error: dataset '{dataset_name}' not found in config.")
        return None, None

    dataset_config = config[dataset_name]
    file_path = dataset_config.get("file_path")
    if not file_path:
        print(f"Error: 'file_path' not defined for dataset '{dataset_name}' in config.yaml or DEFAULT_CONFIG.")
        return None, None

    print(f"[Pipeline] 1. Loading {dataset_name} data...")
    try:
        raw_df = load_data(file_path)
    except Exception as e:
        print(f"Loading error: {e}")
        return None, None
    if raw_df is None:
        print("Loading returned no dataframe.")
        return None, None

    print(f"[Pipeline] 2. Cleaning & Standardizing {dataset_name} data...")
    try:
        cleaned_df = clean_data(raw_df, dataset_config.get("columns", {}))
    except Exception as e:
        print(f"Cleaning error: {e}")
        return None, None
    if cleaned_df is None:
        print("Cleaning failed or returned None.")
        return None, None

    ensure_data_dir()
    cleaned_out = os.path.join(DATA_DIR, f"{dataset_name}_cleaned.csv")
    try:
        cleaned_df.to_csv(cleaned_out, index=False)
    except Exception:
        pass
    cleaned_df_global = cleaned_df
    globals()["cleaned_df_global"] = cleaned_df_global

    print(f"[Pipeline] 3. Transforming {dataset_name} data...")
    try:
        transformed = transform_data(cleaned_df, dataset_config, dataset_name)
    except Exception as e:
        print(f"Transformation error: {e}")
        return cleaned_df, None
    if transformed is None:
        print("Transformation returned None.")
        return cleaned_df, None

    transformed_out = os.path.join(DATA_DIR, f"{dataset_name}_transformed.csv")
    try:
        transformed.to_csv(transformed_out, index=False)
    except Exception:
        pass
    transformed_df = transformed
    globals()["transformed_df"] = transformed_df

    print(f"[Pipeline] 4. Analyzing data & generating initial insights...")
    try:
        insights = analyze_data(transformed, dataset_config)
        if insights:
            print("\nInitial Insights:")
            print(insights)
    except Exception as e:
        print(f"   - Error during analysis: {e}")

    # dataset-specific friendly prints and outputs
    if dataset_name.lower() == "tourism_domestic":
        try:
            tdf = transformed.copy()
            tv = pick_column(tdf, "total_visitors", ["total_visitors", "Total Visitors"]) or "total_visitors"
            dcol = pick_column(tdf, "district", ["district", "District", "district_name"]) or "district"
            if tv in tdf.columns:
                top5 = tdf.sort_values(tv, ascending=False).head(5)
                print("\nTop 5 districts by total visitors (2024):")
                for i, row in enumerate(top5[[dcol, tv]].to_dict(orient="records"), start=1):
                    print(f"  {i}. {row[dcol]}: {int(row[tv]):,}")
                insights_out = os.path.join(DATA_DIR, f"{dataset_name}_insights.csv")
                tdf.sort_values(tv, ascending=False).to_csv(insights_out, index=False)
                print(f"\nTourism insights saved to: {insights_out}")
            # dashboard fallback
            try:
                create_dashboard(tdf.copy(), [tv] if tv in tdf.columns else [])
            except Exception:
                try:
                    import matplotlib.pyplot as plt
                    out_img = os.path.join(DATA_DIR, f"{dataset_name}_top20.png")
                    plot_df = tdf.sort_values(tv, ascending=False).head(20) if tv in tdf.columns else tdf.head(20)
                    if tv in plot_df.columns:
                        plt.figure(figsize=(10, 8))
                        plt.barh(plot_df[dcol].astype(str), plot_df[tv])
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

    if dataset_name.lower() == "consumption_details":
        try:
            cdf = transformed.copy()
            tot_col = pick_column(cdf, "totservices", ["totservices", "total_services"])
            dcol = pick_column(cdf, "division", ["division", "Division", "circle"]) or "division"
            if tot_col and tot_col in cdf.columns:
                top5 = cdf.sort_values(tot_col, ascending=False).head(5)
                print("\nTop 5 rows by total services:")
                for i, row in enumerate(top5[[dcol, tot_col]].to_dict(orient="records"), start=1):
                    print(f"  {i}. {row.get(dcol, 'Unknown')}: {int(row[tot_col]):,}")
                insights_out = os.path.join(DATA_DIR, f"{dataset_name}_insights.csv")
                cdf.sort_values(tot_col, ascending=False).to_csv(insights_out, index=False)
                print(f"\nConsumption insights saved to: {insights_out}")
        except Exception as e:
            print("Error creating consumption-specific insights:", e)

    print("\n[Pipeline] 5. Running predictive analysis (if available)...")
    try:
        if dataset_name.lower() == "health_data":
            predicted_kits, pred_date_kits = predict_future_kits(cleaned_df.copy())
            predicted_high_risk_val, pred_date_hr = predict_high_risk(cleaned_df.copy())
            print(f"   - Predicted MCH kits for {pred_date_kits}: {predicted_kits}")
            print(f"   - Predicted high-risk pregnancies for {pred_date_hr}: {predicted_high_risk_val}")
    except Exception as e:
        print(f"   - Error during prediction: {e}")

    print("\n[Pipeline] 6. Creating geospatial visualization (if available)...")
    try:
        if dataset_name.lower() == "health_data":
            create_choropleth_map(transformed)
        else:
            print("   - Geospatial visualization not configured for this dataset.")
    except Exception as e:
        print(f"   - Error during visualization: {e}")

    print("\n[Pipeline] 7. Creating dashboard visualization (if available)...")
    try:
        if dataset_name.lower() == "health_data":
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
# CLI -> rtgs-cli process helper (optional)
# -----------------------
def _enqueue_output(pipe, queue: Queue):
    for line in iter(pipe.readline, b''):
        queue.put(line.decode(errors='replace'))
    pipe.close()


def run_commands_and_capture(cmds: List[str], timeout: Optional[float] = None) -> str:
    """
    Run rtgs-cli.py as subprocess and feed it commands. Returns combined output text.
    Use with care (only if rtgs-cli.py is available). Non-blocking read with a timeout.
    """
    proc = subprocess.Popen(RTGS_CLI_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    q = Queue()
    t = threading.Thread(target=_enqueue_output, args=(proc.stdout, q), daemon=True)
    t.start()

    out_lines: List[str] = []
    try:
        for c in cmds:
            proc.stdin.write((c + "\n").encode())
            proc.stdin.flush()
            time.sleep(0.2)

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
# Tourism text parser & analysis helper
# ---------------------------
def parse_tourism_domestic_report(text: str) -> Dict[str, Any]:
    res = {"dataset": "tourism_domestic", "year": None, "total_visitors": None, "top_districts": []}
    m_total = re.search(r"In\s+(\d{4}).*combined total of\s*([\d,]+)\s*domestic visitors", text, re.IGNORECASE)
    if m_total:
        res["year"] = int(m_total.group(1))
        res["total_visitors"] = int(m_total.group(2).replace(",", ""))
    pairs = re.findall(r"([A-Za-z0-9\-\s&\.]+?)\s*\(\s*([\d,]+)\s*\)", text)
    if pairs:
        for name, num in pairs:
            name_clean = name.strip().rstrip(",")
            try:
                val = int(num.replace(",", ""))
            except Exception:
                val = None
            res["top_districts"].append((name_clean, val))
    if not res["top_districts"]:
        m_seq = re.search(r"top districts by visitor volume are:\s*([^\n\.]+)", text, re.IGNORECASE)
        if m_seq:
            seq = m_seq.group(1).strip()
            parts = [p.strip() for p in seq.split(",") if p.strip()]
            for p in parts:
                mnum = re.search(r"(.+?)\s*\(\s*([\d,]+)\s*\)", p)
                if mnum:
                    n = mnum.group(1).strip()
                    v = int(mnum.group(2).replace(",", ""))
                    res["top_districts"].append((n, v))
                else:
                    res["top_districts"].append((p, None))
    return res


def format_tourism_domestic_analysis(parsed: Dict[str, Any]) -> str:
    lines = ["=== Tourism Domestic Visitors — Analysis ==="]
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
    lines.append("Recommendations:")
    lines.append("- Consider targeted capacity and marketing actions for the top districts during peak months.")
    lines.append("- Investigate districts with unexpectedly low reporting counts to improve data completeness and quality.")
    lines.append("- Plan seasonal staffing and resource allocation if tourist spikes correlate with service load.")
    return "\n".join(lines)


def run_analysis_for_dataset(dataset: str, report_text: Optional[str] = None):
    ds = dataset.lower()
    if ds == "tourism_domestic":
        if report_text:
            parsed = parse_tourism_domestic_report(report_text)
            print(format_tourism_domestic_analysis(parsed))
            return
        cmds = [f"set_dataset {dataset}", "get_insights total date visitors --year 2024", "get_insights top district visitors --top 10 --year 2024"]
        print("Running CLI commands to compute tourism_domestic analysis...")
        raw = run_commands_and_capture(cmds, timeout=6.0)
        parsed = parse_tourism_domestic_report(raw)
        if parsed.get("total_visitors") is None:
            m_total = re.search(r"Total[^\d]*([\d,]{6,})", raw)
            if m_total:
                parsed["total_visitors"] = int(m_total.group(1).replace(",", ""))
        tops = re.findall(r"Top\s*\d+\s*[:\-]\s*([A-Za-z0-9\-\s&\.]+?)\s*\(\s*([\d,]+)\s*\)", raw)
        if tops:
            parsed["top_districts"] = [(n.strip(), int(v.replace(",", ""))) for n, v in tops]
        print(format_tourism_domestic_analysis(parsed))
    elif ds == "consumption_details":
        # Basic consumption analysis: just run pipeline (preferred) or parse report_text if provided
        if report_text:
            print("Parsing report text for consumption_details is not implemented. Running pipeline instead.")
        cleaned, transformed = run_pipeline("consumption_details")
        if transformed is not None:
            print("Consumption dataset transformed and summary generated.")
        else:
            print("Consumption dataset pipeline failed.")
    else:
        print(f"No dataset-specific analysis implemented for '{dataset}'. Add an implementation in run_analysis_for_dataset().")


# -----------------------
# Interactive CLI helpers
# -----------------------
def read_summary_and_print(ds_name: str):
    path = os.path.join(DATA_DIR, f"{ds_name}_executive_summary.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print("\n" + f.read())
    else:
        print(f"No summary for {ds_name}. Run `set_dataset {ds_name}` to generate it.")


def run_interactive_mode():
    global transformed_df, cleaned_df_global, config, current_dataset_name

    # If transformed_df not loaded, try to load from disk
    if transformed_df is None:
        try:
            transformed_df_local = pd.read_csv(TRANSFORMED_DATA_PATH)
            transformed_df = transformed_df_local  # type: ignore
            globals()["transformed_df"] = transformed_df
            print("Interactive mode started. Transformed data loaded.")
        except FileNotFoundError:
            print("Warning: Transformed data not found. Some commands will require running the pipeline first.")

    print("\nRTGS-CLI started. Type 'help' for available commands. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nRTGS-CLI> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        cmd = user_input.strip()
        lower = cmd.lower()

        if lower in ("exit", "quit"):
            print("Goodbye.")
            break

        if lower == "help":
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

        if lower == "list_metrics":
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline or set_dataset first.")
            else:
                print("Available metrics / columns:")
                print(", ".join([str(c) for c in transformed_df.columns.tolist()]))
            continue

        if lower == "show_summary":
            read_summary_and_print(current_dataset_name)
            continue

        if lower.startswith("set_dataset "):
            parts = cmd.split(" ", 1)
            if len(parts) == 2:
                ds = parts[1].strip()
                if config is None:
                    c = safe_read_yaml(CONFIG_PATH)
                    if c is None:
                        print("No config.yaml found. Using DEFAULT_CONFIG.")
                        globals()["config"] = DEFAULT_CONFIG.copy()
                        globals()["config"] = DEFAULT_CONFIG.copy()
                    else:
                        merged = DEFAULT_CONFIG.copy()
                        merged.update(c)
                        globals()["config"] = merged
                    config_local = globals().get("config", DEFAULT_CONFIG)
                if ds in globals().get("config", {}):
                    current_dataset_name = ds
                    print(f"Switching to dataset '{ds}'. Running pipeline...")
                    cleaned, transformed = run_pipeline(ds)
                    if transformed is not None:
                        try:
                            transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)
                        except Exception:
                            pass
                        globals()["transformed_df"] = transformed
                        globals()["cleaned_df_global"] = cleaned
                        print(f"Dataset switched to '{ds}'.")
                        read_summary_and_print(ds)
                    else:
                        print("Pipeline failed for new dataset.")
                else:
                    print(f"Dataset '{ds}' not found in config.yaml or DEFAULT_CONFIG.")
            else:
                print("Usage: set_dataset <dataset_name>")
            continue

        if lower.startswith("get_insights "):
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline or set_dataset first.")
                continue
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

        if lower == "run_analysis":
            if transformed_df is None:
                print("No transformed data available. Run pipeline first.")
                continue
            print("Running analysis...")
            try:
                insights = analyze_data(transformed_df, config.get(current_dataset_name, {}))
                if insights:
                    print(insights)
            except Exception as e:
                print("Analysis error:", e)
            continue

        if lower.startswith("find_anomalies "):
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline first.")
                continue
            _, metric = cmd.split(" ", 1)
            try:
                anomalies, message = find_anomalies(transformed_df.copy(), metric)
                print(message)
                if anomalies is not None and not anomalies.empty:
                    print(anomalies.to_string(index=False))
            except Exception as e:
                print("Anomaly detection error:", e)
            continue

        if lower.startswith("predict "):
            if cleaned_df_global is None:
                print("No cleaned data available. Run pipeline first.")
                continue
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

        if lower.startswith("root_cause "):
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline first.")
                continue
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

        if lower.startswith("generate_dashboard "):
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline first.")
                continue
            _, rest = cmd.split(" ", 1)
            metrics = [m.strip() for m in rest.split(",") if m.strip()]
            try:
                create_dashboard(transformed_df.copy(), metrics)
                print("Dashboard generated (if supported by dashboard module).")
            except Exception as e:
                print("Dashboard error:", e)
            continue

        if lower.startswith("dashboard_for "):
            if transformed_df is None:
                print("No transformed data loaded. Run pipeline first.")
                continue
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
                        metric_match = metric
                        if transformed_df is not None and metric_match not in transformed_df.columns.tolist():
                            match = difflib.get_close_matches(metric_match, transformed_df.columns.tolist(), n=1, cutoff=0.6)
                            if match:
                                metric_match = match[0]
                        dcol = pick_column(transformed_df, "district", ["district", "districtName", "district_name"]) or "district"
                        res = transformed_df[transformed_df[dcol].astype(str).str.strip().str.title() == district.title()]
                        if not res.empty:
                            print(res[[dcol, metric_match]].to_string(index=False))
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
    c = safe_read_yaml(CONFIG_PATH)
    if c is None:
        print("Warning: config.yaml not found — using built-in DEFAULT_CONFIG for quick runs.")
        c = DEFAULT_CONFIG.copy()
    else:
        merged = DEFAULT_CONFIG.copy()
        merged.update(c)
        c = merged
    config = c
    globals()["config"] = config
    print("Configuration loaded successfully.")

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

    globals()["transformed_df"] = transformed
    globals()["cleaned_df_global"] = cleaned

    print("\nPipeline complete. Starting interactive CLI.")
    run_interactive_mode()


# -----------------------
# CLI entrypoint (argument parsing)
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="main.py controller with analysis support")
    parser.add_argument("--analysis", type=str, help="Run named analysis (e.g., run_analysis)")
    parser.add_argument("--dataset", type=str, help="Dataset name for analysis (e.g., tourism_domestic)")
    parser.add_argument("--report-text", type=str, help="Optional raw report text to parse instead of querying CLI")
    parser.add_argument("--interactive", action="store_true", help="Start pipeline then interactive CLI")
    args = parser.parse_args()

    if args.interactive:
        run_pipeline_and_start_cli()
        return

    if args.analysis == "run_analysis":
        if not args.dataset:
            print("Please provide --dataset <name> for run_analysis.")
            return
        run_analysis_for_dataset(args.dataset, report_text=args.report_text)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
