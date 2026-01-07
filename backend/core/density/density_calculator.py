class DensityCalculator:
    def __init__(self, lane_area_pixels=100000):
        self.lane_area = lane_area_pixels

    def calculate_density(self, detections):
        # Simple density: count of vehicles
        # Advanced: sum of area of bounding boxes / total lane area
        count = len(detections)

        if count < 5:
            return "LOW", count
        elif 5 <= count < 15:
            return "MEDIUM", count
        else:
            return "HIGH", count