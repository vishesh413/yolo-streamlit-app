import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import numpy as np
import time

# -----------------------------------------------------------
# 🎨 Custom CSS + Page Setup
# -----------------------------------------------------------
st.set_page_config(page_title="Student Behavior Detection", page_icon="🎥", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #232526, #414345);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        text-align: center;
        color: #f5f5f5;
        text-shadow: 2px 2px 10px #00000080;
        font-size: 38px;
    }
    .css-1d391kg {
        background-color: #232526 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #ff6a00, #ee0979);
    }
    .stRadio label, .stFileUploader label {
        color: #fff !important;
        font-weight: 600;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        color: #ccc;
        font-size: 15px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 App Title
# -----------------------------------------------------------
st.title("🎥 Student Behavior Detection using YOLO")

st.markdown("""
Welcome to the **AI-based Student Behavior Analysis System** 👨‍🏫  
Detects and classifies student actions in classroom videos or images using **YOLOv8 Deep Learning Model**.  
""")

# -----------------------------------------------------------
# 🚀 Load YOLO Model (cached)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------------------------------------
# 📊 Sidebar Controls
# -----------------------------------------------------------
st.sidebar.title("⚙️ Model Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ About:**\n\nThis system detects student behaviors (e.g., talking, sleeping, using phone) using YOLOv8 trained model.")
st.sidebar.markdown("---")
st.sidebar.info("Developed by Vishesh Dwivedi 💻")

# -----------------------------------------------------------
# 🎛️ Input Selection
# -----------------------------------------------------------
option = st.radio("📁 Choose Input Type:", ["📸 Image", "🎬 Video"])

# -----------------------------------------------------------
# 🖼️ IMAGE Detection
# -----------------------------------------------------------
if option == "📸 Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Detecting objects..."):
            results = model.predict(source=img, conf=confidence)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="✅ Detection Result", use_column_width=True)

            # Show detected classes
            detected_classes = results[0].names
            st.success(f"Detected Classes: {detected_classes}")

# -----------------------------------------------------------
# 🎥 VIDEO Detection
# -----------------------------------------------------------
elif option == "🎬 Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0
        detected_frames = 0
        start_time = time.time()

        with st.spinner("⚙️ Processing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                results = model(frame, conf=confidence)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                if len(results[0].boxes) > 0:
                    detected_frames += 1

            cap.release()

        fps = frame_count / (time.time() - start_time)
        st.success("🎉 Video processing completed!")
        st.write(f"📈 **Frames Processed:** {frame_count}")
        st.write(f"🎯 **Detections Found in:** {detected_frames} frames")
        st.write(f"⚡ **Average FPS:** {fps:.2f}")

# -----------------------------------------------------------
# 🧾 Footer
# -----------------------------------------------------------
st.markdown("---")
st.markdown("<p class='footer'>🚀 Powered by Streamlit & YOLOv8 | Created by Vishesh Dwivedi ❤️</p>", unsafe_allow_html=True)
