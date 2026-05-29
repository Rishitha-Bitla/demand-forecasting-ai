"""
Generate Forecasts — connects XGBoost model to MCP server
===========================================================
This file bridges Sprint 3 (MLflow model) and Sprint 5 (MCP server).
It loads the registered XGBoost model and generates real predictions
for all 80 SKU/warehouse combinations for the next 4 weeks.

Run this file once before starting the MCP server:
    python3 generate_forecasts.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("GENERATING XGBOOST FORECASTS")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Load your data
# ─────────────────────────────────────────
print("\n[STEP 1] Loading sales data...")

df = pd.read_csv('data/sbd_sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sku_id','warehouse','date']).reset_index(drop=True)

print(f"Loaded {len(df):,} rows")

# ─────────────────────────────────────────
# STEP 2 — Build features (same as Sprint 2)
# ─────────────────────────────────────────
print("\n[STEP 2] Building features...")

g = df.groupby(['sku_id','warehouse'])

df['demand_lag1']    = g['demand'].shift(1)
df['demand_lag2']    = g['demand'].shift(2)
df['demand_lag4']    = g['demand'].shift(4)
df['rolling_mean_4'] = g['demand'].transform(
    lambda x: x.shift(1).rolling(4).mean())
df['rolling_mean_8'] = g['demand'].transform(
    lambda x: x.shift(1).rolling(8).mean())
df['rolling_std_4']  = g['demand'].transform(
    lambda x: x.shift(1).rolling(4).std())

df = df.dropna().reset_index(drop=True)

FEATURES = ['week_of_year','year','is_promo',
            'demand_lag1','demand_lag2','demand_lag4',
            'rolling_mean_4','rolling_mean_8','rolling_std_4']

print(f"Features built — {len(df):,} rows ready")

# ─────────────────────────────────────────
# STEP 3 — Load XGBoost model from MLflow
# ─────────────────────────────────────────
print("\n[STEP 3] Loading XGBoost model from MLflow...")

# Point MLflow to where your experiments are stored
mlflow.set_tracking_uri(
    f"sqlite:///notebooks/mlflow.db"
)

try:
    # Load your registered model by name
    model = mlflow.xgboost.load_model(
        "models:/sbd_demand_forecast_xgb/latest"
    )
    print("Model loaded from MLflow registry")
    
except Exception as e:
    print(f"Could not load from registry: {e}")
    print("Falling back to saved pickle model...")
    import pickle
    with open('models/xgb_demand_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from pickle file")

# ─────────────────────────────────────────
# STEP 4 — Generate predictions for all SKUs
# ─────────────────────────────────────────
print("\n[STEP 4] Generating predictions for all SKU/warehouse combinations...")

# Get the most recent data for each SKU and warehouse
# This is what we use to predict the next 4 weeks
latest = df.groupby(['sku_id','warehouse']).tail(1).copy()

# Get last known inventory per SKU per warehouse
inventory = df.groupby(['sku_id','warehouse'])['inventory'].last().reset_index()
inventory.columns = ['sku_id','warehouse','last_inventory']

results = []

for _, row in latest.iterrows():
    sku       = row['sku_id']
    warehouse = row['warehouse']
    
    # Get current inventory
    inv_row = inventory[
        (inventory['sku_id'] == sku) &
        (inventory['warehouse'] == warehouse)
    ]
    current_inventory = int(inv_row['last_inventory'].iloc[0]) if len(inv_row) > 0 else 0
    
    # Generate 4 week forecast
    # Each week uses the previous week's prediction as lag feature
    week_features = row[FEATURES].copy()
    
    weekly_predictions = []
    
    for week in range(1, 5):
        # Get the date for this future week
        future_date = df['date'].max() + pd.Timedelta(weeks=week)
        future_woy  = future_date.isocalendar()[1]
        future_year = future_date.year
        
        # Update time features
        week_features['week_of_year'] = future_woy
        week_features['year']         = future_year
        week_features['is_promo']     = 0  # assume no promo
        
        # Make prediction
        X = pd.DataFrame([week_features[FEATURES]])
        pred = float(model.predict(X)[0])
        pred = max(0, pred)  # demand cannot be negative
        
        weekly_predictions.append(round(pred, 1))
        
        # Update lag features for next week
        week_features['demand_lag4'] = week_features['demand_lag2']
        week_features['demand_lag2'] = week_features['demand_lag1']
        week_features['demand_lag1'] = pred
        
        # Update rolling mean
        week_features['rolling_mean_4'] = (
            week_features['rolling_mean_4'] * 3 + pred
        ) / 4
    
    # Calculate total 4 week forecast
    total_4wk = sum(weekly_predictions)
    avg_weekly = total_4wk / 4
    
    # Calculate weeks of supply based on XGBoost predictions
    weeks_of_supply = round(
        current_inventory / max(avg_weekly, 1), 1
    )
    
    # Risk label based on weeks of supply
    if weeks_of_supply < 16:
        risk = 'HIGH'
    elif weeks_of_supply < 20:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'
    
    results.append({
        'sku_id':            sku,
        'warehouse':         warehouse,
        'current_inventory': current_inventory,
        'week1_forecast':    weekly_predictions[0],
        'week2_forecast':    weekly_predictions[1],
        'week3_forecast':    weekly_predictions[2],
        'week4_forecast':    weekly_predictions[3],
        'total_4wk_forecast': round(total_4wk, 1),
        'avg_weekly_forecast': round(avg_weekly, 1),
        'weeks_of_supply':   weeks_of_supply,
        'risk':              risk,
        'model':             'XGBoost v1 — MLflow registered'
    })

forecasts_df = pd.DataFrame(results)

# ─────────────────────────────────────────
# STEP 5 — Save forecasts to CSV
# ─────────────────────────────────────────
print("\n[STEP 5] Saving forecasts...")

forecasts_df.to_csv('data/xgboost_forecasts.csv', index=False)

print(f"Saved {len(forecasts_df)} forecasts to data/xgboost_forecasts.csv")
print(f"\nRisk breakdown:")
print(forecasts_df['risk'].value_counts().to_string())
print(f"\nSample HIGH risk SKUs:")
high = forecasts_df[forecasts_df['risk']=='HIGH'].head(3)
print(high[['sku_id','warehouse','avg_weekly_forecast',
            'current_inventory','weeks_of_supply','risk']].to_string(index=False))

print("\n" + "=" * 55)
print("FORECASTS GENERATED SUCCESSFULLY")
print("=" * 55)
print("Now restart your MCP server to use XGBoost predictions.")