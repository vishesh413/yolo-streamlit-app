import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🤖", layout="wide")
st.title("🤖 YOLOv8 Object Detection App")
st.markdown("### Upload an Image or Video and let AI detect objects in real-time!")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])

# Load model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")  # 👈 Update this path if needed
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model not loaded: {e}")
        return None

model = load_model()

# Detect function
def detect_objects(image):
    results = model(image, conf=confidence_threshold)
    annotated_frame = results[0].plot()
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    return annotated_frame, list(set(detected_classes))

# Image
if input_type == "Image":
    uploaded_image = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting..."):
                frame, classes = detect_objects(np.array(image))
                st.image(frame, caption="Detected Image", use_container_width=True)
                st.success(f"✅ Detected Objects: {classes if classes else 'No object detected'}")

# Video
else:
    uploaded_video = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)

        if st.button("🎯 Run Detection"):
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture(tfile.name)
                output_path = "output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                )

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)

                frame_no = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=confidence_threshold)
                    annotated_frame = results[0].plot()
                    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    frame_no += 1
                    progress.progress(min(frame_no / frame_count, 1.0))

                cap.release()
                out.release()
                st.video(output_path)
                st.success("✅ Detection completed!")

st.markdown("---")
st.markdown("👨‍💻 Developed by **Vishesh** | YOLOv8 + Streamlit App 🚀")
