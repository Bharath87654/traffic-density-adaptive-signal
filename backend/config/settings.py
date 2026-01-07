import os
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = os.getenv("VIDEO_PATH", "data/raw/traffic_video.mp4")
MODEL_PATH = "models/yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.5
DETECTION_CLASSES = [2, 3, 5, 7]

MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60
VEHICLE_UNIT_TIME = 2