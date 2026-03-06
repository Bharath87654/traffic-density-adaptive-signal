import cv2
from ultralytics import YOLO

class VehicleDetector:

    def __init__(self):

        self.model = YOLO("models/yolov8n.pt")

        self.vehicle_classes = [
            "car",
            "bus",
            "truck",
            "motorcycle"
        ]

    def detect(self, frame):

        results = self.model(frame)[0]

        vehicles = []

        for box in results.boxes:

            cls = int(box.cls[0])
            label = self.model.names[cls]

            if label in self.vehicle_classes:

                x1,y1,x2,y2 = map(int,box.xyxy[0])

                vehicles.append({
                    "type":label,
                    "x":x1,
                    "y":y1,
                    "w":x2-x1,
                    "h":y2-y1
                })

        return vehicles