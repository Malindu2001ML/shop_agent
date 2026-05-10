from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    csv_path: str
    filter_column: Optional[str]
    filter_value: Optional[Any]
    filtered_data: List[Dict]
    crud_result: str
    best_selling: str
    monthly_best: str
    low_stock: str
    complementary: str
    future_prediction: str
    reorder_rec: str
    target_month: int
    target_year: int
    user_question: Optional[str]
    selected_tool: str
    tool_result: str
    llm_answer: str
    final_report: str