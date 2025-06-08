import json
from pathlib import Path
from typing import Any, Dict, Union

from websockets.exceptions import ConnectionClosedOK

from gaf_guard.toolkit.enums import MessageType, Role
from gaf_guard.toolkit.logging import configure_logger


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
        body: Union[None, Any] = None,
        message_type: MessageType = MessageType.PRINT,
        **kwargs,
    ):
        await self.get_websocket(list(self.active_connections.keys())[0]).send(
            json.dumps({"body": body, "message_type": message_type} | kwargs)
        )

    async def log_benchmark(self, data: Any, step: str, role: str, trial_file: str):
        if data:
            if Path(trial_file).exists():
                trial_data = json.loads(Path(trial_file).read_text())
                trial_data.append({"task": step, "role": role, "content": data})
            else:
                trial_data = [{"task": step, "role": role, "content": data}]

            json.dump(trial_data, open(trial_file, "w"), indent=4)

    def disconnect(self, client_id):
        del self.active_connections[client_id]
        logger.info(f"Client disconnected: {client_id}")


conn_manager = WebSocketConnectionManager()
