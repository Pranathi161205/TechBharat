#!/usr/bin/env python3
"""
job_worker.py

Reads jobs/*.json created by ingest_watch.py, extracts column names
from metadata, normalizes them as metrics, and updates catalog.db.
"""

import os, json, re, sqlite3, shutil

BASE = os.path.abspath(os.path.dirname(__file__))
JOBS_DIR = os.path.join(BASE, "jobs")
JOBS_DONE_DIR = os.path.join(JOBS_DIR, "done")
CATALOG_DB = os.path.join(BASE, "catalog", "catalog.db")

os.makedirs(JOBS_DONE_DIR, exist_ok=True)

def normalize_metric(name: str) -> str:
    """Normalize a column name into a metric identifier."""
    s = str(name).strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed_metric"

def process_job(job_path: str):
    print(f"[worker] Processing {job_path}")
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    dataset_id = job["dataset_id"]
    metadata_path = job["metadata_json"]

    if not os.path.exists(metadata_path):
        print(f"[worker] Metadata file missing: {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Extract and normalize metrics
    cols = metadata.get("columns", [])
    metrics = [normalize_metric(c["name"]) for c in cols]

    print(f"[worker] Dataset {dataset_id} metrics: {metrics}")

    # Update catalog
    conn = sqlite3.connect(CATALOG_DB)
    cur = conn.cursor()
    cur.execute("""
        UPDATE datasets
        SET status=?, columns_json=?
        WHERE id=?
    """, ("profiled", json.dumps(metrics), dataset_id))
    conn.commit()
    conn.close()

    # Move job file to done/
    done_path = os.path.join(JOBS_DONE_DIR, os.path.basename(job_path))
    shutil.move(job_path, done_path)
    print(f"[worker] Job {dataset_id} completed -> {done_path}")

def main():
    for fname in os.listdir(JOBS_DIR):
        if fname.endswith(".json"):
            process_job(os.path.join(JOBS_DIR, fname))

if __name__ == "__main__":
    main()
