import asyncio
import cv2

from services.vehicle_detector import VehicleDetector
from services.traffic_state import traffic_state

detector = VehicleDetector()

videos = {
    1:"videos/lane1.mp4",
    2:"videos/lane2.mp4",
    3:"videos/lane3.mp4",
    4:"videos/lane4.mp4"
}

async def detection_socket(websocket):

    await websocket.accept()

    caps = {}

    for lane,path in videos.items():

        caps[lane] = cv2.VideoCapture(path)

    while True:

        for lane,cap in caps.items():

            ret,frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue

            vehicles = detector.detect(frame)

            traffic_state.lanes[lane-1]["vehicles"] = len(vehicles)

            detections = []

            for v in vehicles:

                detections.append({
                    "type":v["type"],
                    "confidence":0.9,
                    "x":v["x"]/frame.shape[1]*100,
                    "y":v["y"]/frame.shape[0]*100,
                    "w":v["w"]/frame.shape[1]*100,
                    "h":v["h"]/frame.shape[0]*100
                })

            await websocket.send_json({
                "lane":lane,
                "detections":detections
            })

        await asyncio.sleep(0.5)