import json
from graph.state import AgentState
from utils.mcp_client import mcp_client

async def agent2_node(state: AgentState) -> AgentState:
    # 1. Read current CSV via CRUD
    crud_res = await mcp_client.call_crud("read_csv", {})
    state["crud_result"] = crud_res

    # 2. Best selling items (all time)
    best = await mcp_client.call_pred("get_best_selling_items", {"limit": 5})
    state["best_selling"] = best

    # 3. Monthly best sellers (from recommendation server)
    month = state.get("target_month", 12)
    year = state.get("target_year", 2010)
    monthly = await mcp_client.call_reco("get_monthly_best_sellers", {"month": month, "year": year, "limit": 5})
    state["monthly_best"] = monthly

    # 4. Low stock items
    low = await mcp_client.call_reco("get_low_stock_items", {"threshold_quantity": 20})
    state["low_stock"] = low

    # 5. Complementary items for top seller
    try:
        bs = json.loads(best)
        top_code = bs["best_selling_items"][0]["StockCode"]
    except:
        top_code = "85123A"
    comp = await mcp_client.call_reco("get_complementary_items", {"stock_code": top_code, "min_lift": 1.2})
    state["complementary"] = comp

    # 6. Future prediction & reorder
    pred = await mcp_client.call_pred("predict_future_demand", {"stock_code": top_code, "days_ahead": 7})
    state["future_prediction"] = pred
    reorder = await mcp_client.call_pred("recommend_reorder_quantity", {"stock_code": top_code, "lead_time": 7, "safety": 1.5})
    state["reorder_rec"] = reorder

    state["final_report"] = f"Filtered: {len(state['filtered_data'])} rows.\nBest_all: {best}\nMonthly: {monthly}\nLow: {low}\nComplementary: {comp}"
    return state