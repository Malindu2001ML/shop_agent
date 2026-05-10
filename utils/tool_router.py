import json
import os
from groq import Groq
from utils.mcp_client import mcp_client
from utils.tool_executor import ToolExecutor

class SmartRouter:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile" 

    async def get_dynamic_tools(self):
        all_tools = []
        server_map = {
            "crud": mcp_client.crud_session,
            "prediction": mcp_client.pred_session,
            "recommendation": mcp_client.reco_session
        }

        for server_name, session in server_map.items():
            if session:
                try:
                    server_tools = await session.list_tools()
                    for tool in server_tools.tools:
                        safe_description = tool.description if tool.description else f"Perform {tool.name} on {server_name}"
                        all_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": safe_description,
                                "parameters": tool.inputSchema
                            },
                            "server": server_name 
                        })
                except Exception as e:
                    print(f"Error listing tools from {server_name}: {e}")
        return all_tools

    async def route_and_execute(self, user_input: str, history=None):
        if history is None: history = []
        
        available_tools_metadata = await self.get_dynamic_tools()
        if not available_tools_metadata:
            return " No tools discovered. Check MCP servers.", "none"

        groq_tools = [{"type": "function", "function": t["function"]} for t in available_tools_metadata]

        messages = [
                {
                    "role": "system", 
                    "content": """You are a Shop Manager. 
                    CRITICAL RULE: Always use the ACTUAL output from the tool. 
                    Do not write your own Python code to 'simulate' the data. 
                    If the tool 'read_csv' returns a result, look inside the 'data' list and find the value.
                    The current date is 2026. Forget all previous hardcoded prices."""
                },
                {"role": "user", "content": user_input}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",
                temperature=0.1
            )
        except Exception as e:
            return f"❌ Groq API Error: {str(e)}", "none"

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return response_message.content, "none"

        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "Error: Invalid tool arguments provided by AI.", "none"

        tool_info = next((t for t in available_tools_metadata if t["function"]["name"] == tool_name), None)
        if not tool_info:
            return f"Error: Tool {tool_name} metadata missing.", "none"

        result = await ToolExecutor.execute(tool_info["server"], tool_name, arguments)
        
        return result, tool_name