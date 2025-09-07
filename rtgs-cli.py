#!/usr/bin/env python3
# debug-friendly mock rtgs-cli for testing rtgs_nlp_wrapper

import sys

PROMPT = "RTGS-CLI> "

def show_help():
    print("Mock RTGS-CLI started. Type 'exit' to quit.")
    print("Available commands: set_dataset, show_summary, run_analysis, get_insights, find_anomalies, predict, root_cause, generate_dashboard, dashboard_for")

def handle(cmd):
    cmd = cmd.strip()
    if not cmd:
        return
    # echo the raw command for debugging
    print(f"[mock received raw command]: '{cmd}'")
    parts = cmd.split()
    c = parts[0]
    args = parts[1:]
    print(f"[mock parsed command='{c}', args={args}]")  # debug line

    if c == 'set_dataset':
        print(f"Dataset set to: {' '.join(args) or 'health_data'}")
    elif c == 'show_summary':
        print("SUMMARY: rows=123, cols=12, districts=10")
    elif c == 'run_analysis':
        print("ANALYSIS: 3 insights found: (1) rising cases in DistrictA, (2) low vaccination in DistrictB, (3) anomaly in malaria cases.")
    elif c == 'get_insights':
        # be robust: if args are empty, try to parse quoted args or fallback sensibly
        if args:
            target = ' '.join(args)
        else:
            target = 'all all'
        print(f"INSIGHTS for {target}: sample insight text...")
    elif c == 'find_anomalies':
        print(f"ANOMALIES detected for {' '.join(args) or 'all'}: sample anomaly at 2025-06")
    elif c == 'predict':
        print(f"PREDICTION: predicted values for {' '.join(args) or 'kits'} (mock).")
    elif c == 'root_cause':
        print(f"ROOT CAUSE analysis for {' '.join(args) or 'all'}: mock causes listed.")
    elif c == 'generate_dashboard':
        print(f"GENERATED dashboard for {' '.join(args) or 'all'} (mock).")
    elif c == 'dashboard_for':
        print(f"DASHBOARD for {' '.join(args)} (mock).")
    elif c in ('exit', 'quit'):
        print("Exiting mock CLI.")
        sys.exit(0)
    else:
        print(f"Unknown command: {cmd}")

def main():
    show_help()
    try:
        while True:
            try:
                cmd = input(PROMPT)
            except EOFError:
                break
            handle(cmd)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
