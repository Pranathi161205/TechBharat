# scripts/predict_data.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def predict_future_kits(df, target_date=None):
    if df is None or df.empty:
        return None, None
    df['dataDate'] = pd.to_datetime(df['dataDate'])
    df.set_index('dataDate', inplace=True)
    monthly_data = df.resample('ME').agg({'kitsCnt': 'sum'})
    monthly_data.reset_index(inplace=True)
    monthly_data['time_stamp'] = monthly_data['dataDate'].apply(lambda x: x.timestamp())
    X = monthly_data[['time_stamp']]
    y = monthly_data['kitsCnt']
    model = LinearRegression()
    model.fit(X, y)
    if target_date:
        prediction_date = pd.to_datetime(target_date)
    else:
        last_date = monthly_data['dataDate'].max()
        prediction_date = last_date + pd.DateOffset(months=1)
    next_month_timestamp = np.array([[prediction_date.timestamp()]])
    predicted_kits = model.predict(next_month_timestamp)[0]
    return round(predicted_kits, 0), prediction_date.strftime('%B %Y')

def predict_high_risk(df, target_date=None):
    if df is None or df.empty:
        return None, None
    df['dataDate'] = pd.to_datetime(df['dataDate'])
    df.set_index('dataDate', inplace=True)
    monthly_data = df.resample('ME').agg({'highRiskCnt': 'sum'})
    monthly_data.reset_index(inplace=True)
    monthly_data['time_stamp'] = monthly_data['dataDate'].apply(lambda x: x.timestamp())
    X = monthly_data[['time_stamp']]
    y = monthly_data['highRiskCnt']
    model = LinearRegression()
    model.fit(X, y)
    if target_date:
        prediction_date = pd.to_datetime(target_date)
    else:
        last_date = monthly_data['dataDate'].max()
        prediction_date = last_date + pd.DateOffset(months=1)
    next_month_timestamp = np.array([[prediction_date.timestamp()]])
    predicted_high_risk = model.predict(next_month_timestamp)[0]
    return round(predicted_high_risk, 0), prediction_date.strftime('%B %Y')