import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image

st.set_page_config(page_title="🎥 Student Behavior Detection", layout="wide")
st.title("🎥 Student Behavior Detection using YOLO")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)
option = st.sidebar.radio("Select Input Type:", ["Image", "Video"])

# File uploader
if option == "Image":
    uploaded_image = st.file_uploader("📸 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Detecting objects..."):
            results = model.predict(source=img, conf=confidence)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Result", use_column_width=True)
        st.success("✅ Detection Complete!")

elif option == "Video":
    uploaded_video = st.file_uploader("🎞️ Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.info("🔁 Processing video, please wait...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        
        cap.release()
        st.success("✅ Video Processing Completed!")
