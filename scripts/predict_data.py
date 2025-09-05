# scripts/predict_data.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def predict_future_kits(df, target_date=None):
    """
    Predicts the total number of kits distributed for a target month
    using a simple linear regression model. If no target_date is provided,
    it predicts for the next month.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for prediction.")
        return None

    # Ensure 'dataDate' is in datetime format and set it as index
    if 'dataDate' not in df.columns:
        print("Error: 'dataDate' column not found in the DataFrame.")
        return None
    
    df['dataDate'] = pd.to_datetime(df['dataDate'])
    df.set_index('dataDate', inplace=True)
    
    # Resample the data to get monthly totals
    monthly_data = df.resample('ME').agg({
        'kitsCnt': 'sum',
        'pwRegCnt': 'sum'
    })
    monthly_data.reset_index(inplace=True)
    
    # Convert dates to a numerical format (e.g., seconds since epoch) for the model
    monthly_data['time_stamp'] = monthly_data['dataDate'].apply(lambda x: x.timestamp())
    
    # Prepare the data for the model
    X = monthly_data[['time_stamp']]
    y = monthly_data['kitsCnt']
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Define the date to predict for
    if target_date:
        prediction_date = pd.to_datetime(target_date)
    else:
        last_date = monthly_data['dataDate'].max()
        prediction_date = last_date + pd.DateOffset(months=1)
        
    next_month_timestamp = np.array([[prediction_date.timestamp()]])
    
    # Predict the value for the next month
    predicted_kits = model.predict(next_month_timestamp)[0]
    
    print("   - Prediction complete.")
    return round(predicted_kits, 0), prediction_date.strftime('%B %Y')