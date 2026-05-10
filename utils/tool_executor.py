from datetime import datetime
from utils.mcp_client import mcp_client

class ToolExecutor:
    @staticmethod
    async def execute(server: str, tool_name: str, parameters: dict) -> str:
        """Dispatches the tool call to the correct MCP client session."""
        
        # Map server names to their respective client calling methods
        dispatch_map = {
            "crud": mcp_client.call_crud,
            "prediction": mcp_client.call_pred,
            "recommendation": mcp_client.call_reco
        }

        if server not in dispatch_map:
            return f"Error: Server '{server}' is not recognized."

        # Adaptive Logic for 'create_row'
        if tool_name == "create_row":
            # Groq might send row_data nested or flat; we normalize it
            row_data = parameters.get("row_data", parameters)
            
            # Final validation before hitting the CSV
            required = ["StockCode", "Description", "Quantity"]
            missing = [field for field in required if field not in row_data]
            
            if missing:
                return f"I need more details to add this item. Missing: {', '.join(missing)}"

            # Set smart defaults
            row_data.setdefault("InvoiceNo", f"AI-{datetime.now().strftime('%S%M%H')}")
            row_data.setdefault("InvoiceDate", datetime.now().strftime("%Y-%m-%d %H:%M"))
            row_data.setdefault("UnitPrice", 0.0)
            row_data.setdefault("CustomerID", 0)
            
            parameters = {"row_data": row_data}

        # Call the actual MCP tool via the client
        try:
            result = await dispatch_map[server](tool_name, parameters)
            return result
        except Exception as e:
            return f"Execution Error on {server}: {str(e)}"