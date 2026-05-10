import pandas as pd
from graph.state import AgentState

async def agent1_node(state: AgentState) -> AgentState:
    # Read CSV with proper encoding
    df = pd.read_csv(state["csv_path"], encoding='latin-1')
    col = state.get("filter_column")
    val = state.get("filter_value")
    if col and val is not None and col in df.columns:
        df = df[df[col] == val]
    state["filtered_data"] = df.to_dict(orient="records")
    print(f"[Agent1] Filtered {len(state['filtered_data'])} rows")
    return state