from fastapi import WebSocket
import asyncio
from services.traffic_state import traffic_state

async def signal_socket(websocket: WebSocket):

    await websocket.accept()

    while True:

        await websocket.send_json({
            "active_lane": traffic_state.active_lane,
            "countdown": traffic_state.countdown
        })

        await asyncio.sleep(1)

        traffic_state.countdown -= 1

        if traffic_state.countdown <= 0:

            traffic_state.active_lane += 1

            if traffic_state.active_lane > 4:
                traffic_state.active_lane = 1

            traffic_state.countdown = 15