from mcp.server.fastmcp import FastMCP
import pandas as pd
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(base_dir, "data", "shop_data.csv")
RULES_PATH = os.path.join(base_dir, "data", "models", "association_rules.pkl")

mcp = FastMCP("Shop Recommendation Server")

rules = None
if os.path.exists(RULES_PATH):
    rules = joblib.load(RULES_PATH)

@mcp.tool()
def get_monthly_best_sellers(month: int, year: int = 2010, limit: int = 10) -> dict:
    df = pd.read_csv(CSV_PATH, encoding='latin-1')   
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    mask = (df["InvoiceDate"].dt.month == month) & (df["InvoiceDate"].dt.year == year)
    month_df = df[mask]
    if month_df.empty:
        return {"error": f"No data for {year}-{month}"}
    grouped = month_df.groupby(["StockCode","Description"])["Quantity"].sum().reset_index()
    grouped = grouped.sort_values("Quantity", ascending=False)
    return {"month": month, "year": year, "best_sellers": grouped.head(limit).to_dict(orient="records")}

@mcp.tool()
def get_low_stock_items(threshold_quantity: int = 20) -> dict:
    df = pd.read_csv(CSV_PATH, encoding='latin-1')   # FIXED
    grouped = df.groupby(["StockCode","Description"])["Quantity"].sum().reset_index()
    low = grouped[grouped["Quantity"] < threshold_quantity]
    return {"low_stock_items": low.to_dict(orient="records")}

@mcp.tool()
def get_complementary_items(stock_code: str, min_lift: float = 1.2) -> dict:
    if rules is None:
        return {"error": "Run training/train_association_rules.py first"}
    relevant = rules[rules['antecedents'].apply(lambda x: stock_code in x)]
    relevant = relevant[relevant['lift'] >= min_lift]
    recs = []
    for _, row in relevant.iterrows():
        for cons in row['consequents']:
            if cons != stock_code:
                recs.append({"item": cons, "confidence": row['confidence'], "lift": row['lift']})
    unique = {}
    for r in recs:
        it = r["item"]
        if it not in unique or r["lift"] > unique[it]["lift"]:
            unique[it] = r
    return {"stock_code": stock_code, "complementary_items": list(unique.values())}

if __name__ == "__main__":
    mcp.run()