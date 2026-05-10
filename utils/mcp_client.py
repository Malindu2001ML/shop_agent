import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = AsyncExitStack()
            cls._instance.crud_session = None
            cls._instance.pred_session = None
            cls._instance.reco_session = None
        return cls._instance

    async def connect(self):
        # CRUD server
        crud_params = StdioServerParameters(command="python", args=["mcp_servers/crud_server.py"])
        crud_ctx = stdio_client(crud_params)
        crud_read, crud_write = await self.stack.enter_async_context(crud_ctx)
        self.crud_session = await self.stack.enter_async_context(ClientSession(crud_read, crud_write))
        await self.crud_session.initialize()

        # Prediction server
        pred_params = StdioServerParameters(command="python", args=["mcp_servers/prediction_server.py"])
        pred_ctx = stdio_client(pred_params)
        pred_read, pred_write = await self.stack.enter_async_context(pred_ctx)
        self.pred_session = await self.stack.enter_async_context(ClientSession(pred_read, pred_write))
        await self.pred_session.initialize()

        # Recommendation server
        reco_params = StdioServerParameters(command="python", args=["mcp_servers/recommendation_server.py"])
        reco_ctx = stdio_client(reco_params)
        reco_read, reco_write = await self.stack.enter_async_context(reco_ctx)
        self.reco_session = await self.stack.enter_async_context(ClientSession(reco_read, reco_write))
        await self.reco_session.initialize()

    async def call_crud(self, tool: str, args: dict) -> str:
        result = await self.crud_session.call_tool(tool, args)
        return self._extract_text(result)

    async def call_pred(self, tool: str, args: dict) -> str:
        result = await self.pred_session.call_tool(tool, args)
        return self._extract_text(result)

    async def call_reco(self, tool: str, args: dict) -> str:
        result = await self.reco_session.call_tool(tool, args)
        return self._extract_text(result)

    def _extract_text(self, result):
        if hasattr(result, 'content') and result.content:
            texts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                else:
                    texts.append(str(item))
            return " ".join(texts)
        return "{}"

    async def close(self):
        await self.stack.aclose()

mcp_client = MCPClientManager()