from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.lanes import router as lanes_router
from api.analytics import router as analytics_router
from api.signal import router as signal_router

from websocket.signal_ws import signal_socket
from websocket.detection_ws import detection_socket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(lanes_router)
app.include_router(analytics_router)
app.include_router(signal_router)

@app.websocket("/ws/signal-status")
async def signal_ws(websocket: WebSocket):

    await signal_socket(websocket)

@app.websocket("/ws/vehicle-detection")
async def detection_ws(websocket: WebSocket):

    await detection_socket(websocket)