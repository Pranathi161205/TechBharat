# run_root_tourism.py
import pandas as pd
from scripts.analyze_data import run_root_cause_analysis

# path to transformed CSV
path = "data/tourism_domestic_transformed.csv"
df = pd.read_csv(path)

print("Available columns:", df.columns.tolist())
metric = "total_visitors"   # change to avg_monthly_visitors if you prefer

if metric not in df.columns:
    raise SystemExit(f"Metric '{metric}' not found in {path}")

print(f"Running root-cause analysis for: {metric}\n")
res = run_root_cause_analysis(df.copy(), metric)
print(res)
