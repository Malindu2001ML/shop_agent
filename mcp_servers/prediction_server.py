from fastmcp import FastMCP
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

CSV_PATH = "data/shop_data.csv"
MODEL_PATH = "data/models/demand_model.pkl"
SCALER_PATH = "data/models/scaler.pkl"
FEATURE_COLS_PATH = "data/models/feature_cols.pkl"

mcp = FastMCP("ML Demand Prediction Server")

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Run training/train_demand_model.py first")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, cols

try:
    model, scaler, feature_cols = load_artifacts()
except:
    model = scaler = feature_cols = None

@mcp.tool()
def predict_future_demand(stock_code: str, days_ahead: int = 7) -> dict:
    if model is None:
        return {"error": "Model not trained yet"}
    df = pd.read_csv(CSV_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    item_df = df[df["StockCode"] == stock_code].copy()
    if item_df.empty:
        return {"error": f"StockCode '{stock_code}' not found"}
    item_df["DateOnly"] = item_df["InvoiceDate"].dt.date
    daily = item_df.groupby("DateOnly")["Quantity"].sum().reset_index()
    daily["DateOnly"] = pd.to_datetime(daily["DateOnly"])
    daily = daily.sort_values("DateOnly").reset_index(drop=True)
    last_date = daily["DateOnly"].iloc[-1]
    last_q = daily["Quantity"].iloc[-1]
    lag1 = daily["Quantity"].iloc[-2] if len(daily)>1 else last_q
    lag2 = daily["Quantity"].iloc[-3] if len(daily)>2 else last_q
    roll = daily["Quantity"].rolling(3).mean().iloc[-1] if len(daily)>=3 else last_q

    preds = []
    c_lag1, c_lag2, c_roll = last_q, lag1, roll
    for i in range(days_ahead):
        future = last_date + timedelta(days=i+1)
        feats = np.array([[
            future.weekday(), future.month, future.day,
            c_lag1, c_lag2, c_roll
        ]])
        scaled = scaler.transform(feats)
        p = model.predict(scaled)[0]
        preds.append(max(0, round(p,2)))
        c_lag2, c_lag1 = c_lag1, p
        c_roll = (c_roll*2 + p)/3 if i>0 else (c_roll+p)/2
    return {
        "stock_code": stock_code,
        "daily_predictions": preds,
        "total_predicted_demand": round(sum(preds),2)
    }

@mcp.tool()
def get_best_selling_items(limit: int = 5) -> dict:
    df = pd.read_csv(CSV_PATH)
    grouped = df.groupby(["StockCode","Description"])["Quantity"].sum().reset_index()
    grouped = grouped.sort_values("Quantity", ascending=False)
    return {"best_selling_items": grouped.head(limit).to_dict(orient="records")}

@mcp.tool()
def recommend_reorder_quantity(stock_code: str, lead_time: int = 7, safety: float = 1.5) -> dict:
    pred = predict_future_demand(stock_code, lead_time)
    if "error" in pred:
        return pred
    total = pred["total_predicted_demand"]
    return {
        "stock_code": stock_code,
        "recommended_order": round(total * safety),
        "lead_time_days": lead_time,
        "safety_factor": safety
    }

if __name__ == "__main__":
    mcp.run()