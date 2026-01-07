import streamlit as st
import cv2
import tempfile
import time

from core.detection.vehicle_detector import VehicleDetector
from signal_control.signal_logic import SignalController


@st.cache_resource
def load_assets():
    return VehicleDetector(), SignalController()


st.set_page_config(page_title="Smart Traffic AI", layout="wide")
st.title("🚦 Smart Traffic AI – Adaptive Signal Control")

detector, controller = load_assets()

st.sidebar.header("System Controls")

line_pos = st.sidebar.slider(
    "Detection Line Position (Y)",
    min_value=0,
    max_value=540,
    value=450
)

conf_val = st.sidebar.slider(
    "AI Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4
)

frame_skip = st.sidebar.slider(
    "Frame Skip (Performance vs Accuracy)",
    min_value=1,
    max_value=5,
    value=2,
    help="1 = Highest accuracy, 5 = Highest speed"
)

if "run" not in st.session_state:
    st.session_state.run = False

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:

    # Reset counters if new video is uploaded
    new_video = (
        "last_file" not in st.session_state
        or st.session_state.last_file != uploaded_file.name
    )

    if st.sidebar.button("Reset Statistics") or new_video:
        detector.vehicle_count = 0
        detector.crossed_ids.clear()
        detector.prev_positions.clear()
        st.session_state.last_file = uploaded_file.name

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    col1, col2, col3 = st.columns(3)
    flow_metric = col1.empty()
    dens_metric = col2.empty()
    time_metric = col3.empty()

    video_container = st.empty()

    # Start / Stop buttons
    start_col, stop_col = st.columns(2)

    if start_col.button("▶ Start Analysis"):
        st.session_state.run = True

    if stop_col.button("⏹ Stop"):
        st.session_state.run = False
        video_container.empty()
        st.stop()

    if st.session_state.run:
        st.success("System running (GPU accelerated if available)")
    else:
        st.warning("System paused")

    # =================================================
    # 5. PROCESSING LOOP (OPTIMIZED)
    # =================================================
    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.session_state.run = False
            break

        st.session_state.frame_count += 1

        # Resize frame for speed
        frame = cv2.resize(frame, (640, 360))
        detector.line_y = line_pos

        # -------------------------------------------------
        # FRAME SKIPPING LOGIC
        # -------------------------------------------------
        if st.session_state.frame_count % frame_skip == 0:
            detections, total_passed = detector.process_frame(
                frame, conf_val
            )
            st.session_state.last_detections = detections
            st.session_state.last_total = total_passed
        else:
            detections = st.session_state.get("last_detections", [])
            total_passed = st.session_state.get(
                "last_total", detector.vehicle_count
            )

        green_time = controller.get_adaptive_timing(len(detections))

        # =================================================
        # VISUALIZATION
        # =================================================
        cv2.line(
            frame,
            (0, line_pos),
            (frame.shape[1], line_pos),
            (0, 255, 255),
            2
        )

        for d in detections:
            x1, y1, x2, y2 = map(int, d["box"])
            cx, cy = d["centroid"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID:{d['id']}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        # Update Streamlit UI
        video_container.image(
            frame,
            channels="BGR",
            use_container_width=True,
            clamp=True
        )

        flow_metric.metric("Vehicles Passed (Flow)", total_passed)
        dens_metric.metric("Active Density", len(detections))
        time_metric.metric("Green Signal Time", f"{green_time}s")

        # Throttle UI updates
        time.sleep(0.03)

    cap.release()
