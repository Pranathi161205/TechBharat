# api_server.py
"""
RTGS FastAPI wrapper

Endpoints:
  GET  /health
  POST /pipeline/run?dataset=<name>
  GET  /metrics?dataset=<name>
  GET  /insights?dataset=<name>&group=<group>&metric=<metric>
  POST /analysis/run?dataset=<name>
  GET  /summary?dataset=<name>
  POST /nlp_query   -> body: {"query":"...", "dataset_hint":"optional"}
"""

import os
import sys
import subprocess
import importlib
import json
import datetime
import re
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Body, Header, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

import pandas as pd

# Optional: requests for internal HTTP fallback calls
try:
    import requests
except Exception:
    requests = None  # not required if we can import main.py directly

# Optional: import NLP agent if available
try:
    from nlp_agent import parse_nl_query  # parse_nl_query(text, dataset_hint=None) -> dict
except Exception:
    parse_nl_query = None

# Configuration
API_KEY = os.environ.get("RTGS_API_KEY", "")  # set to non-empty string to enable header auth
MAIN_PY = "main.py"
DATA_DIR = "data"
TRANSFORMED_SUFFIX = "_transformed.csv"
SUMMARY_SUFFIX = "_executive_summary.txt"
LOG_NLP_QUERIES = os.path.join(DATA_DIR, "nlp_queries.log")

app = FastAPI(title="RTGS Pipeline API", version="0.2")


# -----------------------
# Utilities
# -----------------------
def _check_api_key(x_api_key: Optional[str]):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header")


def _transformed_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}{TRANSFORMED_SUFFIX}")


def _summary_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}{SUMMARY_SUFFIX}")


def _read_transformed(dataset: str) -> pd.DataFrame:
    p = _transformed_path(dataset)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def shutil_which_python() -> str:
    # prefer current interpreter
    return sys.executable or "python"


def _safe_call_main_cmd(args: List[str], timeout: int = 120) -> str:
    """
    Call main.py as a subprocess with args list (e.g. ['--analysis','run_analysis','--dataset','x']).
    Returns combined stdout/stderr. This is a fallback if importing main.py fails.
    """
    cmd = [shutil_which_python(), MAIN_PY] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    try:
        # stream output until completion or timeout
        for line in proc.stdout:
            out_lines.append(line)
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    return "".join(out_lines)


def log_nl_query(query: str, parse: Dict[str, Any], result: Any):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        entry = {"time": datetime.datetime.utcnow().isoformat(), "query": query, "parse": parse, "result": result}
        with open(LOG_NLP_QUERIES, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # don't fail the request just because logging failed
        pass


def _coerce_number(v):
    """Try to coerce v to int if possible, else float, else None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        iv = int(v)
        return iv
    except Exception:
        try:
            fv = float(str(v).replace(",", ""))
            return fv
        except Exception:
            return None


def _sum_numeric_from_rows(rows: List[dict], key: str):
    total = 0
    found = False
    for r in rows:
        if key in r:
            num = _coerce_number(r.get(key))
            if num is not None:
                total += num
                found = True
    return (total if found else None)


def _extract_year_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return m.group(1)
    return None


# -----------------------
# Basic endpoints
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "data_dir_exists": os.path.isdir(DATA_DIR)}


@app.post("/pipeline/run")
def run_pipeline(dataset: str = Query(..., description="Dataset name as in config.yaml"),
                 x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)

    # Try importing main.run_pipeline directly (fast, in-process)
    try:
        main_mod = importlib.import_module("main")
        if hasattr(main_mod, "run_pipeline"):
            cleaned, transformed = main_mod.run_pipeline(dataset, interactive=False)
            if transformed is None:
                return JSONResponse(status_code=500, content={"error": "Pipeline failed or returned no transformed DataFrame"})
            # try to save transformed to data/
            try:
                os.makedirs(DATA_DIR, exist_ok=True)
                transformed.to_csv(_transformed_path(dataset), index=False)
            except Exception:
                pass
            return {"status": "ok", "dataset": dataset, "rows": len(transformed)}
    except Exception as e:
        # fallback to subprocess call
        try:
            out = _safe_call_main_cmd(["--analysis", "run_analysis", "--dataset", dataset], timeout=180)
            return {"status": "ok", "dataset": dataset, "output": out}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Failed to run pipeline: {e} / {e2}")


@app.get("/metrics")
def list_metrics(dataset: str = Query(..., description="Dataset name"), x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    try:
        df = _read_transformed(dataset)
        return {"columns": df.columns.tolist()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transformed dataset not found. Run /pipeline/run first.")


@app.get("/insights")
def get_insights(dataset: str = Query(...), group: str = Query(...), metric: str = Query(...),
                 x_api_key: Optional[str] = Header(None)):
    """
    Lookup rows where detected grouping column == group and return the requested metric.
    """
    _check_api_key(x_api_key)
    try:
        df = _read_transformed(dataset)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transformed dataset not found. Run /pipeline/run first.")

    # detect grouping column (prefer common ones)
    candidates = ["district", "division", "group", "circle", "area", "subdivision", "name", "region", "state"]
    gcol = next((c for c in candidates if c in df.columns), None)
    if gcol is None:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        gcol = text_cols[0] if text_cols else df.columns[0]

    # tolerant matching: substring, case-insensitive, optional year filter with fallback
    group_norm = (group or "").strip()
    if not group_norm:
        mask = pd.Series([False] * len(df), index=df.index)
    else:
        mask = df[gcol].astype(str).fillna("").str.contains(re.escape(group_norm), case=False, na=False)

    year = _extract_year_from_text(group)  # here group may contain year in some callers; no harm
    year_cols = [c for c in df.columns if c.lower() in ("year", "yr", "date", "fiscal_year")]
    if year and year_cols:
        ycol = year_cols[0]
        try:
            mask_year = df[ycol].astype(str).str.contains(str(year), na=False)
            mask = mask & mask_year
            if not mask.any():
                ys = df[ycol].astype(str).dropna().unique().tolist()
                years_found = []
                for v in ys:
                    m = re.search(r"(20\d{2})", str(v))
                    if m:
                        years_found.append(int(m.group(1)))
                if years_found:
                    latest = str(max(years_found))
                    mask = df[gcol].astype(str).fillna("").str.contains(re.escape(group_norm), case=False, na=False) & df[ycol].astype(str).str.contains(latest, na=False)
        except Exception:
            pass

    res = df.loc[mask]
    if res.empty:
        sample_vals = df[gcol].dropna().astype(str).unique()[:20].tolist()
        return JSONResponse(status_code=404, content={"error": f"No rows found for {group} in '{gcol}'", "sample_values": sample_vals})

    if metric not in df.columns:
        return JSONResponse(status_code=404, content={"error": f"Metric '{metric}' not found", "available": df.columns.tolist()})

    return {"group_column": gcol, "rows": res[[gcol, metric]].to_dict(orient="records")}


@app.post("/analysis/run")
def run_analysis(dataset: str = Query(...), x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    # Try to call run_analysis_for_dataset in main.py
    try:
        main_mod = importlib.import_module("main")
        if hasattr(main_mod, "run_analysis_for_dataset"):
            # capture printed output
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                main_mod.run_analysis_for_dataset(dataset)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            return {"status": "ok", "dataset": dataset, "output": output}
    except Exception as e:
        # fallback to subprocess
        try:
            out = _safe_call_main_cmd(["--analysis", "run_analysis", "--dataset", dataset], timeout=120)
            return {"status": "ok", "dataset": dataset, "output": out}
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Failed to run analysis: {e} / {ex}")


@app.get("/summary")
def get_summary(dataset: str = Query(...), x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    p = _summary_path(dataset)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Summary file not found. Run the pipeline first.")
    return FileResponse(p, media_type="text/plain", filename=os.path.basename(p))


# -----------------------
# NLP endpoint
# -----------------------
@app.post("/nlp_query")
def nlp_query(payload: Dict[str, Any] = Body(...), x_api_key: Optional[str] = Header(None)):
    """
    Body: {"query": "Show top 5 divisions by billed services in Hyderabad", "dataset_hint": "consumption_details"}
    Returns parse + mapped action and action result (best-effort).
    """
    _check_api_key(x_api_key)

    query = payload.get("query") if isinstance(payload, dict) else None
    dataset_hint = payload.get("dataset_hint") if isinstance(payload, dict) else None
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Provide JSON body like {'query': '...'}")

    # Parse the NL query using nlp_agent if available, otherwise use a simple fallback
    parse = None
    try:
        if parse_nl_query:
            parse = parse_nl_query(query, dataset_hint=dataset_hint)
        else:
            # Simple fallback parser: rudimentary extraction using keywords and regex
            # (keeps behavior safe even if nlp_agent missing)
            text = query.lower()
            parse = {"intent": None, "dataset": None, "group": None, "metric": None, "n": None, "raw": query}
            # basic intent detection
            if "top" in text or "highest" in text or re.search(r"top\s+\d+", text):
                parse["intent"] = "top_n"
            elif "summary" in text:
                parse["intent"] = "summary"
            elif "run" in text and "pipeline" in text:
                parse["intent"] = "run_pipeline"
            else:
                parse["intent"] = "get_insights"
            # basic dataset hinting
            if dataset_hint:
                parse["dataset"] = dataset_hint
            elif "health" in text or "kit" in text or "anc" in text:
                parse["dataset"] = "health_data"
            elif "tourism" in text or "visitor" in text:
                parse["dataset"] = "tourism_domestic"
            elif "temperature" in text or "temp" in text:
                parse["dataset"] = "temperature_data"
            elif "consumption" in text or "billed" in text or "services" in text:
                parse["dataset"] = "consumption_details"
            # group extraction "in <place>" or "for <place>"
            m = re.search(r"(?:in|for|at)\s+([A-Za-z0-9\-\s&,]+)", query, re.IGNORECASE)
            if m:
                parse["group"] = m.group(1).strip()
            # metric basic guess
            if "billed" in text or "bill" in text:
                parse["metric"] = "total_billed_services"
            if "service" in text and not parse.get("metric"):
                parse["metric"] = "total_services"
            if "visitor" in text and not parse.get("metric"):
                parse["metric"] = "total_visitors"
            if "kit" in text and "coverage" in text:
                parse["metric"] = "kit_coverage_ratio"
            if "high risk" in text or "highrisk" in text:
                parse["metric"] = "high_risk_ratio"
            if "avg temp" in text or "average temperature" in text:
                parse["metric"] = "avg_temp"
            if "max temp" in text or "maximum temperature" in text:
                parse["metric"] = "max_temperature"
            # n extraction
            m2 = re.search(r"top\s+(\d+)", query, re.IGNORECASE)
            if m2:
                parse["n"] = int(m2.group(1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parser failure: {e}")

    # ensure dataset chosen
    dataset = parse.get("dataset") or dataset_hint
    intent = parse.get("intent")
    group = parse.get("group")
    metric = parse.get("metric")
    n = parse.get("n") or 5

    # Ensure pipeline ran (best-effort) -- try in-process first
    try:
        main_mod = importlib.import_module("main")
        if hasattr(main_mod, "run_pipeline"):
            try:
                main_mod.run_pipeline(dataset, interactive=False)
            except Exception:
                # ignore pipeline errors here; continue to try reading transformed file if present
                pass
    except Exception:
        # fallback: try HTTP call to our own /pipeline/run (if requests available)
        if requests:
            try:
                requests.post(f"http://127.0.0.1:8000/pipeline/run?dataset={dataset}", timeout=30)
            except Exception:
                pass

    # Map intent -> action
    try:
        if intent in ("get_insights", None):
            if not group or not metric:
                result = {"note": "missing group or metric; provide a clearer query or include dataset_hint"}
                log_nl_query(query, parse, result)
                return {"parse": parse, "result": result}

            df = _read_transformed(dataset)
            # detect grouping column like /insights
            candidates = ["district", "division", "group", "circle", "area", "subdivision"]
            gcol = next((c for c in candidates if c in df.columns), None)
            if gcol is None:
                text_cols = [c for c in df.columns if df[c].dtype == object]
                gcol = text_cols[0] if text_cols else df.columns[0]
            mask = df[gcol].astype(str).str.strip().str.title() == group.strip().title()
            rows = df.loc[mask, [gcol, metric]].to_dict(orient="records")
            result = {"group_column": gcol, "rows": rows}
            log_nl_query(query, parse, result)
            return {"parse": parse, "result": result}

        elif intent == "top_n":
            if not metric:
                result = {"note": "metric not found in parse"}
                log_nl_query(query, parse, result)
                return {"parse": parse, "result": result}
            df = _read_transformed(dataset)
            if metric not in df.columns:
                raise HTTPException(status_code=404, detail=f"Metric '{metric}' not present in transformed dataset.")
            top_df = df.sort_values(metric, ascending=False).head(n)
            result = {"top": top_df.to_dict(orient="records")}
            log_nl_query(query, parse, result)
            return {"parse": parse, "result": result}

        elif intent == "summary":
            p = _summary_path(dataset)
            if not os.path.exists(p):
                raise HTTPException(status_code=404, detail="Summary not found. Run pipeline first.")
            with open(p, "r", encoding="utf-8") as fh:
                txt = fh.read()
            result = {"summary": txt}
            log_nl_query(query, parse, result)
            return {"parse": parse, "result": result}

        elif intent == "run_pipeline":
            try:
                main_mod = importlib.import_module("main")
                if hasattr(main_mod, "run_pipeline"):
                    cleaned, transformed = main_mod.run_pipeline(dataset, interactive=False)
                    result = {"status": "pipeline_run_complete", "rows": len(transformed) if transformed is not None else 0}
                    log_nl_query(query, parse, result)
                    return {"parse": parse, "result": result}
            except Exception:
                # fallback to subprocess
                out = _safe_call_main_cmd(["--analysis", "run_analysis", "--dataset", dataset], timeout=180)
                result = {"status": "pipeline_run_subprocess", "output": out}
                log_nl_query(query, parse, result)
                return {"parse": parse, "result": result}

        else:
            result = {"note": "Intent not implemented"}
            log_nl_query(query, parse, result)
            return {"parse": parse, "result": result}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Transformed dataset not found. Run pipeline first.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
