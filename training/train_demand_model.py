import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/shop_data.csv"
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_features(df, stock_code):
    df = df[df["StockCode"] == stock_code].copy()
    if df.empty:
        return None, None, None
    
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    df = df.sort_values("InvoiceDate")
    df["DateOnly"] = df["InvoiceDate"].dt.date
    daily = df.groupby("DateOnly")["Quantity"].sum().reset_index()
    daily["DateOnly"] = pd.to_datetime(daily["DateOnly"])
    daily = daily.sort_values("DateOnly")
    daily["day_of_week"] = daily["DateOnly"].dt.dayofweek
    daily["month"] = daily["DateOnly"].dt.month
    daily["day_of_month"] = daily["DateOnly"].dt.day
    daily["lag_1"] = daily["Quantity"].shift(1)
    daily["lag_2"] = daily["Quantity"].shift(2)
    daily["rolling_mean_3"] = daily["Quantity"].rolling(3).mean()
    daily.fillna(0, inplace=True)
    features = ["day_of_week","month","day_of_month","lag_1","lag_2","rolling_mean_3"]
    X = daily[features].values
    y = daily["Quantity"].values
    return X, y, features

def train_model(stock_code):
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    X, y, feats = create_features(df, stock_code)
    if X is None or len(X) < 5:
        raise ValueError(f"Not enough data for {stock_code}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, os.path.join(MODEL_DIR, "demand_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feats, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print(f" Trained demand model for {stock_code} with {len(X)} daily samples")

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    top_stock = df.groupby("StockCode")["Quantity"].sum().idxmax()
    train_model(top_stock)