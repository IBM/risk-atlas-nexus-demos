from typing import Literal, Union, Dict
import json
from agentic_governance.toolkit.logging import configure_logger
from websockets.exceptions import ConnectionClosedOK
from agentic_governance.toolkit.enums import MessageType

logger = configure_logger(__name__)


class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections = {}

    def get_websocket(self, client_id):
        return self.active_connections[client_id]

    async def accept(self, websocket):
        client_id = websocket.request.headers["client_id"]
        self.active_connections[client_id] = websocket
        logger.info(f"Client connected: {client_id}")
        return client_id

    async def receive(self, client_id):
        try:
            async for message in self.get_websocket(client_id):
                yield json.loads(message) | {"client_id": client_id}
        except ConnectionClosedOK:
            self.disconnect(client_id)

    async def send(
        self,
        body: Union[None, str] = None,
        message_type: Literal[MessageType.RULE, MessageType.PRINT] = MessageType.PRINT,
        **kwargs,
    ):
        await self.get_websocket(list(self.active_connections.keys())[0]).send(
            json.dumps({"message_type": message_type, "body": body} | kwargs)
        )

    async def log(
        self,
        body: Dict,
        workflow_step: str,
        **kwargs,
    ):
        if hasattr(body, "get") and body.get("log", None):
            await self.send(body["log"], workflow_step=workflow_step, **kwargs)

    async def completed(
        self,
        client_id: str,
    ):
        await self.send(message_type=MessageType.COMPLETED)

    def disconnect(self, client_id):
        del self.active_connections[client_id]
        logger.info(f"Client disconnected: {client_id}")


conn_manager = WebSocketConnectionManager()
