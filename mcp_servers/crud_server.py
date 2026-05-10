from fastmcp import FastMCP
import pandas as pd
import os

CSV_PATH = "data/shop_data.csv"
REQUIRED_COLS = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID"]

mcp = FastMCP("CSV CRUD Server")

def ensure_csv():
    if not os.path.exists(CSV_PATH):
        empty_df = pd.DataFrame(columns=REQUIRED_COLS)
        empty_df.to_csv(CSV_PATH, index=False)

@mcp.tool()
def read_csv() -> dict:
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    return df.to_dict(orient="records")

@mcp.tool()
def create_row(row_data: dict) -> dict:
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    for col in REQUIRED_COLS:
        if col not in row_data:
            row_data[col] = None
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return {"success": True, "row": row_data}

@mcp.tool()
def update_item_by_code(stock_code: str, new_data: dict) -> dict:
    """
    Updates an item's details based on its StockCode. 
    Use this to change Price, Quantity, or Description.
    """
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    
    df['StockCode'] = df['StockCode'].astype(str)
    stock_code_str = str(stock_code)

    if stock_code_str not in df['StockCode'].values:
        return {"error": f"Item {stock_code} not found."}

    for col, value in new_data.items():
        if col in df.columns:
            df.loc[df['StockCode'] == stock_code_str, col] = value

    df.to_csv(CSV_PATH, index=False)
    return {"success": True, "message": f"Item {stock_code} updated successfully."}

@mcp.tool()
def delete_item_by_code(stock_code: str) -> dict:
    """
    Deletes an item from the CSV based on its unique StockCode.
    This is safer than using an index.
    """
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    
 
    df['StockCode'] = df['StockCode'].astype(str)
    stock_code_str = str(stock_code)

    if stock_code_str not in df['StockCode'].values:
        return {"error": f"Item with StockCode {stock_code} not found in database."}

    df = df[df['StockCode'] != stock_code_str]
    df.to_csv(CSV_PATH, index=False)
    
    return {"success": True, "message": f"Item {stock_code} has been deleted."}

@mcp.tool()
def delete_row(index: int) -> dict:
    ensure_csv()
    df = pd.read_csv(CSV_PATH)
    if index < 0 or index >= len(df):
        return {"error": "Index out of range"}
    df = df.drop(index).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    return {"success": True, "deleted_index": index}

if __name__ == "__main__":
    ensure_csv()
    mcp.run()