#!/usr/bin/env python3
"""
ingest_watch.py

Simple ingestion watcher:
- monitors data/incoming/ for new CSV files
- moves file to staging, computes checksum
- sniffs delimiter/encoding, reads sample rows
- normalizes column names
- writes canonical CSV + parquet, metadata JSON
- registers dataset in SQLite catalog and emits a job file
"""

import os
import time
import shutil
import hashlib
import json
import sqlite3
import uuid
import csv
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List

import pandas as pd  # requires pandas
# pandas.to_parquet requires pyarrow or fastparquet
# pip install pandas pyarrow

# -----------------------
# Configuration
# -----------------------
BASE = os.path.abspath(os.path.dirname(__file__))
DATA_RAW = os.path.join(BASE, "data", "raw")
DATA_STAGING = os.path.join(DATA_RAW, "staging")
DATA_INCOMING = os.path.join(BASE, "data", "incoming")
DATA_PARQUET = os.path.join(BASE, "data", "parquet")
CATALOG_DIR = os.path.join(BASE, "catalog")
JOBS_DIR = os.path.join(BASE, "jobs")
CATALOG_DB = os.path.join(CATALOG_DIR, "catalog.db")

POLL_INTERVAL = 5.0
SAMPLE_ROWS = 1000
MAX_AUTOPROCESS_FILESIZE = 500 * 1024 * 1024  # 500MB default limit, change if needed

# -----------------------
# Helpers
# -----------------------
def ensure_dirs():
    for d in (DATA_INCOMING, DATA_STAGING, DATA_PARQUET, CATALOG_DIR, JOBS_DIR):
        os.makedirs(d, exist_ok=True)

def sha256_of_file(path: str, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_dataset_id(original_name: str) -> str:
    # Create dataset id: sanitized filename (no ext) + uuid4 short
    name = os.path.splitext(os.path.basename(original_name))[0]
    name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name).strip('_').lower()
    return f"{name}__{uuid.uuid4().hex[:8]}"

def detect_delimiter_and_encoding(sample_path: str) -> Tuple[str, str]:
    # Try to guess delimiter with csv.Sniffer on a byte sample, test encodings
    probable_delimiters = [",", "\t", ";", "|"]
    enc_candidates = ["utf-8", "latin1", "utf-8-sig"]
    sample_bytes = None
    with open(sample_path, "rb") as f:
        sample_bytes = f.read(65536)
    for enc in enc_candidates:
        try:
            txt = sample_bytes.decode(enc, errors="strict")
            sn = csv.Sniffer()
            dialect = sn.sniff(txt, delimiters="".join(probable_delimiters))
            return dialect.delimiter, enc
        except Exception:
            continue
    # fallback
    return ",", "utf-8"

def normalize_column(col: str) -> str:
    if col is None:
        return ""
    s = str(col).strip().lower()
    s = re.sub(r'[^\w\s\-]', '', s)  # drop punctuation except underscore/hyphen
    s = s.replace(" ", "_")
    s = re.sub(r'__+', '_', s)
    s = s.strip('_')
    if not s:
        s = "col"
    return s

def profile_dataframe(df: pd.DataFrame, sample_n: int = 10) -> List[Dict[str, Any]]:
    cols = []
    for c in df.columns:
        ser = df[c]
        dtype = str(ser.dtype)
        sample_vals = ser.dropna().astype(str).unique()[:sample_n].tolist()
        null_pct = float(ser.isna().sum()) / max(1, len(ser))
        col_info = {
            "name": c,
            "dtype": dtype,
            "null_pct": round(null_pct, 4),
            "sample_values": sample_vals,
            "n_unique_sample": int(ser.nunique())
        }
        cols.append(col_info)
    return cols

# -----------------------
# Catalog (SQLite)
# -----------------------
def init_catalog(db_path: str = CATALOG_DB):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        original_filename TEXT,
        checksum TEXT,
        csv_path TEXT,
        parquet_path TEXT,
        metadata_json_path TEXT,
        created_at TEXT,
        status TEXT,
        row_count INTEGER,
        columns_json TEXT
    )
    """)
    conn.commit()
    conn.close()

def register_dataset(record: Dict[str, Any], db_path: str = CATALOG_DB):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO datasets (id, original_filename, checksum, csv_path, parquet_path, metadata_json_path, created_at, status, row_count, columns_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["id"],
        record["original_filename"],
        record["checksum"],
        record["csv_path"],
        record["parquet_path"],
        record["metadata_json_path"],
        record["created_at"],
        record["status"],
        record.get("row_count", 0),
        json.dumps(record.get("columns", []))
    ))
    conn.commit()
    conn.close()

# -----------------------
# Process a single file
# -----------------------
def process_file(incoming_path: str):
    print(f"[ingest] Found file: {incoming_path}")
    # Quick size check
    filesize = os.path.getsize(incoming_path)
    if filesize > MAX_AUTOPROCESS_FILESIZE:
        print(f"[ingest] File too large ({filesize} bytes). Move to quarantine or handle externally.")
        return

    # Move to staging (atomic move)
    basename = os.path.basename(incoming_path)
    staging_name = f"{int(time.time())}__{basename}"
    staging_path = os.path.join(DATA_STAGING, staging_name)
    shutil.move(incoming_path, staging_path)
    print(f"[ingest] Moved to staging: {staging_path}")

    checksum = sha256_of_file(staging_path)
    print(f"[ingest] SHA256: {checksum}")

    # Detect delimiter and encoding
    delimiter, encoding = detect_delimiter_and_encoding(staging_path)
    print(f"[ingest] Detected delimiter='{delimiter}' encoding='{encoding}'")

    # Read a sample to infer schema
    try:
        df_sample = pd.read_csv(staging_path, sep=delimiter, encoding=encoding, nrows=SAMPLE_ROWS)
    except Exception as e:
        print(f"[ingest] Error reading CSV sample: {e}. Attempting fallback with utf-8-sig.")
        try:
            df_sample = pd.read_csv(staging_path, sep=delimiter, encoding="utf-8-sig", nrows=SAMPLE_ROWS)
        except Exception as e2:
            print(f"[ingest] Failed to read CSV: {e2}. Moving to quarantine.")
            quarantine_dir = os.path.join(DATA_RAW, "quarantine")
            os.makedirs(quarantine_dir, exist_ok=True)
            shutil.move(staging_path, os.path.join(quarantine_dir, basename))
            return

    # Normalize columns
    original_columns = list(df_sample.columns)
    normalized_cols = [normalize_column(c) for c in original_columns]
    # If normalized col names clash, make them unique
    seen = {}
    final_cols = []
    for c in normalized_cols:
        base = c or "col"
        if base in seen:
            seen[base] += 1
            c2 = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            c2 = base
        final_cols.append(c2)

    # Read full dataframe streaming-safe (pandas will handle memory for moderate files)
    try:
        df = pd.read_csv(staging_path, sep=delimiter, encoding=encoding, names=original_columns if df_sample.shape[0] and df_sample.columns.equals(pd.Index(original_columns)) else None, low_memory=False)
        # rename columns to normalized
        df.columns = final_cols
    except Exception as e:
        print(f"[ingest] Error reading full CSV: {e}. Attempting to read with engine='python'")
        try:
            df = pd.read_csv(staging_path, sep=delimiter, encoding=encoding, engine="python", low_memory=False)
            df.columns = final_cols
        except Exception as e2:
            print(f"[ingest] Failed to read CSV full: {e2}. Quarantining.")
            quarantine_dir = os.path.join(DATA_RAW, "quarantine")
            os.makedirs(quarantine_dir, exist_ok=True)
            shutil.move(staging_path, os.path.join(quarantine_dir, basename))
            return

    # Prepare dataset id and folders
    dataset_id = safe_dataset_id(basename)
    dataset_folder = os.path.join(DATA_RAW, dataset_id)
    os.makedirs(dataset_folder, exist_ok=True)

    csv_dest = os.path.join(dataset_folder, f"{dataset_id}__v1.csv")
    parquet_dest = os.path.join(DATA_PARQUET, f"{dataset_id}__v1.parquet")
    metadata_json = os.path.join(dataset_folder, "metadata__v1.json")

    # Move staging file to canonical csv path (keep a copy)
    shutil.move(staging_path, csv_dest)
    print(f"[ingest] Stored canonical CSV: {csv_dest}")

    # Save parquet
    try:
        df.to_parquet(parquet_dest, index=False)
        print(f"[ingest] Wrote parquet: {parquet_dest}")
    except Exception as e:
        print(f"[ingest] Failed to write parquet: {e}. You may need pyarrow/fastparquet installed.")

    # Profile
    columns_profile = profile_dataframe(df.head(SAMPLE_ROWS))

    # Metadata
    record = {
        "id": dataset_id,
        "original_filename": basename,
        "checksum": checksum,
        "csv_path": csv_dest,
        "parquet_path": parquet_dest,
        "metadata_json_path": metadata_json,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "ingested",
        "row_count": int(len(df)),
        "columns": columns_profile
    }

    # Write metadata JSON
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Wrote metadata: {metadata_json}")

    # Register in catalog
    try:
        register_dataset(record)
        print(f"[ingest] Registered dataset in catalog: {dataset_id}")
    except Exception as e:
        print(f"[ingest] Failed to register dataset in catalog: {e}")

    # Emit job file for downstream processors
    job = {
        "dataset_id": dataset_id,
        "csv_path": csv_dest,
        "parquet_path": parquet_dest,
        "metadata_json": metadata_json,
        "created_at": record["created_at"]
    }
    job_path = os.path.join(JOBS_DIR, f"{dataset_id}.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    print(f"[ingest] Emitted job: {job_path}")

# -----------------------
# Watch loop
# -----------------------
def main():
    ensure_dirs()
    init_catalog()
    print("[ingest] Watching folder:", DATA_INCOMING)
    # track files currently being processed (by checksum) to avoid double-processing
    processing = set()
    try:
        while True:
            # list CSV-like files in incoming
            for fname in os.listdir(DATA_INCOMING):
                if not fname.lower().endswith((".csv", ".txt")):
                    continue
                inpath = os.path.join(DATA_INCOMING, fname)
                # skip partial files: check size stable for a short time
                try:
                    size1 = os.path.getsize(inpath)
                except FileNotFoundError:
                    continue
                time.sleep(0.2)
                try:
                    size2 = os.path.getsize(inpath)
                except FileNotFoundError:
                    continue
                if size1 != size2:
                    # still writing
                    continue
                # compute checksum quickly (or defer until moved to staging)
                try:
                    csum = sha256_of_file(inpath)
                except Exception:
                    csum = None
                if csum and csum in processing:
                    continue
                # mark and process
                if csum:
                    processing.add(csum)
                try:
                    process_file(inpath)
                except Exception as e:
                    print(f"[ingest] Error processing file {inpath}: {e}")
                finally:
                    if csum and csum in processing:
                        processing.discard(csum)
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("[ingest] Interrupted. Exiting.")

if __name__ == "__main__":
    main()
