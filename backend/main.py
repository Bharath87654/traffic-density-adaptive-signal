import cv2
from core.detection.vehicle_detector import VehicleDetector
from signal_control.signal_logic import SignalController
from config import settings


def main():
    cap = cv2.VideoCapture(settings.VIDEO_PATH)
    detector = VehicleDetector()
    controller = SignalController()

    ret, dummy_frame = cap.read()
    if ret:
        detector.process_frame(cv2.resize(dummy_frame, (960, 540)))

    print("--- System Initialized & Optimized ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (960, 540))

        detections, total_passed = detector.process_frame(frame)

        suggested_time = controller.get_adaptive_timing(len(detections))

        cv2.line(frame, (0, detector.line_y), (960, detector.line_y), (0, 255, 255), 2)

        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            # Draw Bounding Box & ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{det['id']}", (x1, y1 - 5), 0, 0.5, (0, 255, 0), 1)

        # Dashboard Overlay
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Flow (Total): {total_passed}", (20, 40), 2, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Density (Now): {len(detections)}", (20, 70), 2, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"Green Timer: {suggested_time}s", (20, 100), 2, 0.7, (0, 255, 0), 2)

        cv2.imshow("Production-Ready Traffic System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()