import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="YOLO Object Detection", page_icon="🤖", layout="wide")

# Header
st.title("🤖 Real-Time Object Detection App")
st.markdown("### Powered by **YOLOv8 + Streamlit** 🚀")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
source_option = st.sidebar.radio("Select Input Source", ("📸 Upload Image", "🎥 Upload Video"))

# Load Model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Main Detection Function
def detect_objects(image):
    results = model(image, conf=confidence_threshold)
    annotated_frame = results[0].plot()
    return annotated_frame, results

# Main Section
if source_option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting..."):
                annotated_frame, results = detect_objects(np.array(image))
                st.image(annotated_frame, caption="Detected Image", use_container_width=True)
                st.success("✅ Detection Complete!")

elif source_option == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())
        st.video("temp_video.mp4")

        if st.button("🎯 Run Detection on Video"):
            with st.spinner("Processing Video..."):
                cap = cv2.VideoCapture("temp_video.mp4")
                out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
                                      int(cap.get(cv2.CAP_PROP_FPS)),
                                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=confidence_threshold)
                    annotated_frame = results[0].plot()
                    out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                cap.release()
                out.release()
                st.video("output.mp4")
                st.success("✅ Video Detection Complete!")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by [Vishesh] 💫 | YOLOv8 + Streamlit")

