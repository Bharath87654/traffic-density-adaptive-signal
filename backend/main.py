import streamlit as st
import cv2
import tempfile
import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from core.detection.vehicle_detector import VehicleDetector
from signal_control.signal_logic import SignalController

# -------------------------------------------------
# 1. PAGE CONFIG & ASSET LOADING
# -------------------------------------------------
st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    layout="wide",
    page_icon="🚦"
)

CSV_FILE = "vehicles.csv"
os.makedirs("analytics", exist_ok=True)


@st.cache_resource
def load_assets():
    return VehicleDetector(), SignalController()


detector, controller = load_assets()


# -------------------------------------------------
# 2. LOGIN LOGIC
# -------------------------------------------------
def authenticate(username, password):
    return username == "admin" and password == "password"


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.stop()

# -------------------------------------------------
# 3. SESSION STATE INIT
# -------------------------------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "execution_data" not in st.session_state:
    st.session_state.execution_data = {"all_vehicle_ids": set()}
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# -------------------------------------------------
# 4. SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("🎛 Controls")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.divider()
conf_val = st.sidebar.slider("AI Confidence Threshold", 0.1, 1.0, 0.35)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 2)
clearance_rate = st.sidebar.number_input("Seconds per Vehicle", value=2.5)

uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

# -------------------------------------------------
# 5. HEADER & TABS
# -------------------------------------------------
st.title("🚦 Traffic Density Analysis & Adaptive Signal Control")
st.caption("AI-based real-time traffic monitoring system")
st.divider()

tab1, tab2, tab3 = st.tabs(["🚘 Live Traffic", "📊 Analytics", "🧠 System Info"])

# ======================================================
# TAB 1: LIVE TRAFFIC
# ======================================================
with tab1:
    if not uploaded_file:
        st.info("Please upload a video file in the sidebar to begin analysis.")
    else:
        col_video, col_metrics = st.columns([2, 1])

        with col_video:
            st.subheader("📹 Traffic Feed")
            video_placeholder = st.empty()
            c1, c2 = st.columns(2)
            start_btn = c1.button("▶ Start Analysis", use_container_width=True)
            stop_btn = c2.button("⏹ Stop & Save", use_container_width=True)

        with col_metrics:
            st.subheader("📌 Live Metrics")
            dens_m = st.metric("🚗 Current Frame Density", "0")
            time_m = st.metric("⏱ Clearance Time", "0s")
            signal_status = st.empty()

        if start_btn:
            st.session_state.run = True
            st.session_state.execution_data["all_vehicle_ids"].clear()
            st.session_state.frame_count = 0

        if stop_btn:
            st.session_state.run = False
            total_unique = len(st.session_state.execution_data["all_vehicle_ids"])
            total_signal_time = total_unique * clearance_rate

            log_entry = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Total_Vehicles": total_unique,
                "Signal_Time_Sec": round(total_signal_time, 2)
            }
            pd.DataFrame([log_entry]).to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
            st.success(f"✅ Data Logged! Total Unique Vehicles: {total_unique}")

        if st.session_state.run:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            while cap.isOpened() and st.session_state.run:
                ret, frame = cap.read()
                if not ret: break

                st.session_state.frame_count += 1

                # DETECTION: Process raw frame for accuracy
                if st.session_state.frame_count % frame_skip == 0:
                    detections, _ = detector.process_frame(frame, conf_val)
                    for det in detections:
                        st.session_state.execution_data["all_vehicle_ids"].add(det["id"])

                    cur_count = len(detections)
                    dens_m.metric("🚗 Current Frame Density", cur_count)
                    time_m.metric("⏱ Clearance Time", f"{round(cur_count * clearance_rate, 1)}s")

                    # Signal Visual Status
                    if cur_count > 15:
                        signal_status.error("🔴 RED – High Density")
                    elif cur_count > 5:
                        signal_status.warning("🟡 YELLOW – Medium Density")
                    else:
                        signal_status.success("🟢 GREEN – Low Density")

                    st.session_state.last_detections = detections
                else:
                    detections = st.session_state.get("last_detections", [])

                # Resize ONLY for UI display
                display_frame = cv2.resize(frame, (854, 480))
                for d in detections:
                    x1, y1, x2, y2 = map(int, d["box"])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01)
            cap.release()

# ======================================================
# TAB 2: ANALYTICS
# ======================================================
with tab2:
    st.subheader("📊 Traffic Data Visualizations")
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)

        col1, col2 = st.columns(2)
        with col1:
            st.write("🚗 Total Vehicle Count Per Run")
            # Correcting column name mapping from your CSV structure
            st.line_chart(df.set_index("Time")["Total_Vehicles"])

        with col2:
            st.write("📊 Traffic Volume Distribution")
            fig, ax = plt.subplots()
            # Simple aggregation for pie chart
            labels = ['Processed Vehicles']
            sizes = [df['Total_Vehicles'].sum()]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig)

        st.divider()
        st.write("### 📜 Execution History")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No analytics data available yet. Run an analysis to generate reports.")

# ======================================================
# TAB 3: SYSTEM INFO
# ======================================================
with tab3:
    st.subheader("🏗 System Architecture")

    st.markdown("""
    **Process Flow:**
    1. **Capture Traffic Video**: Uploaded footage is ingested via OpenCV.
    2. **Frame Extraction**: Performance optimization via frame-skipping.
    3. **Vehicle Detection**: YOLO-based unique ID tracking ensures precise counting.
    4. **Density Classification**: Real-time analysis of vehicles per frame.
    5. **Adaptive Signal Control**: Calculation of clearing time based on density.
    """)

    st.subheader("🚀 Future Enhancements")
    st.markdown("""
    • Emergency vehicle priority (Ambulance/Fire Engine detection)  
    • AI-based traffic prediction for urban planning  
    • Smart city IoT integration  
    """)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("Developed for Traffic Density Analysis & Adaptive Signal Control v2.5")