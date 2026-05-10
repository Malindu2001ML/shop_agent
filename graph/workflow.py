from langgraph.graph import StateGraph, END
from agents.agent1 import agent1_node
from agents.agent2 import agent2_node
from agents.llm_agent import get_llm_agent
from utils.tool_router import SmartRouter
from graph.state import AgentState

async def tool_router_node(state: AgentState) -> AgentState:
    
    if not state.get("user_question"):
        state["tool_result"] = "No question provided."
        state["selected_tool"] = "none"
        return state

    # Initialize the SmartRouter
    router = SmartRouter()
    
 
    result, actual_tool_name = await router.route_and_execute(state["user_question"])

    # Store the execution result in the state
    state["tool_result"] = str(result)
    
    # Instead of a static string, we use the real tool name (e.g., 'delete_row')
    state["selected_tool"] = actual_tool_name 
    
    return state

async def llm_node(state: AgentState) -> AgentState:
    """
    Final node that interprets the tool output and provides a natural language answer.
    """
    if not state.get("user_question"):
        state["llm_answer"] = "No question asked."
        return state

    # Safeguard against missing data in state
    tool_res = state.get('tool_result', 'No result available')
    tool_used = state.get('selected_tool', 'none')

    # Combining tool results with precomputed state data for rich context
    context = f"""
Actual Tool Used: {tool_used}
Tool Execution Result: {tool_res}

Additional Business Intelligence:
- Items in current filter: {len(state.get('filtered_data', []))}
- Top selling products: {state.get('best_selling', 'N/A')}
- Critical low stock items: {state.get('low_stock', 'N/A')}
- Market Basket Insights: {state.get('complementary', 'N/A')}
- Demand Forecast: {state.get('future_prediction', 'N/A')}
- Recommended Reorder Quantity: {state.get('reorder_rec', 'N/A')}
"""
    
    agent = get_llm_agent()
    answer = await agent.answer(state["user_question"], context)
    
    state["llm_answer"] = answer
    
    # Append to the final session report using the correct state variables
    state["final_report"] += (
        f"\n\n--- Step: Smart Tool Execution ---\n"
        f"User Query: {state['user_question']}\n"
        f"Tool Used: {tool_used}\n"
        f"Execution Outcome: {tool_res}\n"
        f"AI Insight: {answer}"
    )
    return state

def build_graph():
    """
    Builds and compiles the multi-agent LangGraph workflow.
    """
    graph = StateGraph(AgentState)
    
    # Define Nodes
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("tool_router", tool_router_node)
    graph.add_node("llm", llm_node)
    
    # Define Edges (Sequential Flow)
    graph.set_entry_point("agent1")
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "tool_router")
    graph.add_edge("tool_router", "llm")
    graph.add_edge("llm", END)
    
    return graph.compile()