

python shop_multi_agent/mcp_servers/crud_server.py &
python shop_multi_agent/mcp_servers/prediction_server.py &
python shop_multi_agent/mcp_servers/recommendation_server.py &

sleep 5

streamlit run shop_multi_agent/ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0