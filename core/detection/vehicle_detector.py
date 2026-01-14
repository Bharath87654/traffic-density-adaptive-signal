import torch
from ultralytics import YOLO
import ultralytics
import functools

# --- PYTORCH 2.6 SECURITY BYPASS ---
# This forces torch to load weights without the strict security check
# that causes the UnpicklingError on Streamlit Cloud.
torch.load = functools.partial(torch.load, weights_only=False)


class VehicleDetector:
    def __init__(self):
        # Determine device (Streamlit Cloud uses CPU, local PC uses CUDA)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load both models from the models/ directory
        # If yolov8n.pt was deleted from GitHub, it will auto-download now
        self.traffic_model = YOLO('models/yolov8n.pt')
        self.emergency_model = YOLO('models/emergency_best.pt')

        self.traffic_model.to(self.device)
        self.emergency_model.to(self.device)

    def process_frame(self, frame, conf_threshold):
        # 1. Track Normal Traffic using the UI slider confidence
        traffic_results = self.traffic_model.track(
            frame, persist=True, conf=conf_threshold, verbose=False, device=self.device
        )

        # 2. Detect Emergency Vehicles using a FIXED high confidence (0.75)
        emergency_results = self.emergency_model(frame, conf=0.75, verbose=False, device=self.device)

        current_detections = []
        is_emergency = False

        # Parse Normal Traffic
        if traffic_results and traffic_results[0].boxes.id is not None:
            boxes = traffic_results[0].boxes.xyxy.cpu().numpy()
            ids = traffic_results[0].boxes.id.cpu().numpy().astype(int)
            clss = traffic_results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):
                # Filter for: car(2), motorcycle(3), bus(5), truck(7)
                if cls in [2, 3, 5, 7]:
                    current_detections.append({"box": box, "id": obj_id, "type": "normal"})

        # Parse Emergency Vehicles with strict class verification
        if len(emergency_results[0].boxes) > 0:
            for box in emergency_results[0].boxes:
                # Assuming class 0 is your trained emergency vehicle class
                if box.cls == 0:
                    is_emergency = True
                    current_detections.append({
                        "box": box.xyxy.cpu().numpy()[0],
                        "id": -1,
                        "type": "emergency"
                    })

        return current_detections, is_emergency