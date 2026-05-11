import subprocess

import streamlit as st
import asyncio
import pandas as pd
import json
import sys, os
import time
from pathlib import Path
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

@st.cache_resource
def start_mcp_servers():
    processes = []
    
    server_scripts = [
        os.path.join(root_dir, "mcp_servers", "crud_server.py"),
        os.path.join(root_dir, "mcp_servers", "prediction_server.py"),
        os.path.join(root_dir, "mcp_servers", "recommendation_server.py")
    ]
    
    st.write("Starting MCP Servers...")
    
    for script in server_scripts:
        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(process)
        time.sleep(1) 
        
    return processes

if 'servers_started' not in st.session_state:
    st.session_state.mcp_processes = start_mcp_servers()
    st.session_state.servers_started = True
    st.success("All 3 MCP Servers are running in the background!")

# --- Path & Env Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from graph.workflow import build_graph
from utils.mcp_client import mcp_client

# --- UI Configuration ---
st.set_page_config(page_title="Smart Shop Analytics Terminal", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 10px; }
    div.stButton > button:first-child {
        background-color: #00cc66;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Dashboard Logic ---
def show_dashboard(csv_file):
    if not os.path.exists(csv_file):
        st.warning("CSV file not found. Please ensure the path is correct and the file exists.")
        return

    try:
        df = pd.read_csv(csv_file, sep=None, engine='python', on_bad_lines='skip')
        display_cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
        df_filtered = df[[c for c in display_cols if c in df.columns]]

        st.divider()
        st.subheader("📊 Live Inventory Database")
        st.dataframe(df_filtered, use_container_width=True, height=400)
            
    except Exception as e:
        st.error(f"Dashboard not displaying: {e}")

# --- Header ---
st.title("🛍️ Smart Shop Multi-Agent Terminal")
st.caption("Real-time Inventory Management with AI Decision Making")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    csv_path = st.text_input("CSV Database Path", "data/shop_data.csv")
    if st.button("🔄 Manual Sync"):
        st.rerun()

show_dashboard(csv_path)

user_question = st.chat_input("Ex: Add 10 units of 'Power Bank' for 25.00 each")

if user_question:
    async def run_workflow():
        try:
            await mcp_client.connect()
            state = {
                "csv_path": csv_path, "filter_column": None, "filter_value": None,
                "filtered_data": [], "crud_result": "", "best_selling": "",
                "monthly_best": "", "low_stock": "", "complementary": "",
                "future_prediction": "", "reorder_rec": "", 
                "target_month": 5, "target_year": 2026,
                "user_question": user_question, "selected_tool": "", "tool_result": "",
                "llm_answer": "", "final_report": ""
            }
            app = build_graph()
            return await app.ainvoke(state)
        finally:
            await mcp_client.close()

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is processing your request..."):
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                final = loop.run_until_complete(run_workflow())

                st.markdown("### 🤖 Agent Response")
                st.info(final.get("llm_answer", "change the data."))
                
                m1, m2 = st.columns(2)
                m1.metric("Selected Agent", final.get("selected_tool", "None"))
                m2.metric("Status", " Task Completed")

               
                st.write("---")
                st.warning("⚠️ Note: The dashboard will not auto-refresh. Please click 'Update Table Now' to see changes reflected in the inventory table below.")
                if st.button("📥 Update Table Now"):
                    st.rerun()

                # --- Logs ---
                with st.expander("🛠 View Execution Log"):
                    st.code(final.get("final_report", ""))

            except Exception as e:
                st.error(f"Error: {e}")