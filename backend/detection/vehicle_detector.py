import cv2
from ultralytics import YOLO
from detection.lane_manager import LaneManager


class VehicleDetector:

    def __init__(self):

        self.model = YOLO("models/yolov8n.pt")
        self.lane_manager = LaneManager()

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        results = self.model(frame, conf=0.35, verbose=False)

        self.lane_manager.reset()

        if results and results[0].boxes is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, cls in zip(boxes, classes):

                if cls in [2, 3, 5, 7]:

                    x1, y1, x2, y2 = box

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    lane = self.lane_manager.assign_lane(center_x, center_y, width, height)

                    self.lane_manager.lanes[lane] += 1

        return self.lane_manager.lanes