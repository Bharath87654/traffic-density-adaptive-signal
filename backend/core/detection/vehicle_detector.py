import cv2
import torch
from ultralytics import YOLO
from config import settings


class VehicleDetector:
    def __init__(self):

        self.model = YOLO(settings.MODEL_PATH)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.line_y = 450
        self.vehicle_count = 0
        self.crossed_ids = set()
        self.prev_positions = {}

    def process_frame(self, frame, conf_threshold):

        results = self.model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=0.5,
            tracker="bytetrack.yaml",
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )

        current_detections = []

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):

                if cls not in settings.DETECTION_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                prev_y = self.prev_positions.get(obj_id, cy)

                if prev_y < self.line_y <= cy:
                    if obj_id not in self.crossed_ids:
                        self.vehicle_count += 1
                        self.crossed_ids.add(obj_id)

                self.prev_positions[obj_id] = cy

                current_detections.append({
                    "box": [x1, y1, x2, y2],
                    "id": obj_id,
                    "centroid": (cx, cy)
                })

        return current_detections, self.vehicle_count
