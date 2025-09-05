# main.py

import os
import pandas as pd
from scripts.clean_data import clean_data, load_data
from scripts.transform_data import transform_data
from scripts.analyze_data import analyze_data # Import the new function

# Define file paths
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data.csv')
CLEANED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_cleaned.csv')
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'mch_kit_data_transformed.csv')

def run_pipeline():
    """
    Main function to run the data processing pipeline.
    """
    print("[Policymaker CLI] --> [RTGS Agent]")
    print("1. Loading Health Dataset...")
    raw_df = load_data(RAW_DATA_PATH)
    if raw_df is None:
        return

    print("2. Cleaning & Standardizing Data...")
    cleaned_df = clean_data(raw_df)
    if cleaned_df is None:
        return
    print(f"   - Original rows: {len(raw_df)}")
    print(f"   - Cleaned rows: {len(cleaned_df)}")
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"   - Cleaned data saved to {CLEANED_DATA_PATH}")

    print("\n3. Transforming Data...")
    transformed_df = transform_data(cleaned_df)
    if transformed_df is None:
        return
    transformed_df.to_csv(TRANSFORMED_DATA_PATH, index=False)
    print(f"   - Transformed data saved to {TRANSFORMED_DATA_PATH}")

    # Step 4: Analyze the data and generate insights
    print("\n4. Analyzing Data & Generating Insights...")
    insights = analyze_data(transformed_df)
    if insights:
        print("\n\n5. Outputting Insights:")
        print(insights)

if __name__ == '__main__':
    run_pipeline()