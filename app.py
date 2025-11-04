import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

# -------------------------------
# 🎯 Page config
# -------------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🧠 YOLO Object Detection App")
st.markdown("Upload an **image** or **video** and let your custom YOLO model detect objects!")

# -------------------------------
# ⚙️ Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"  # Your trained model file
    model = YOLO(model_path)
    return model

model = load_model()

# -------------------------------
# 🎛️ Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
source_type = st.sidebar.radio("Select Source Type:", ("Image", "Video"))

# -------------------------------
# 📸 Image Detection
# -------------------------------
def detect_image(uploaded_file):
    image = Image.open(uploaded_file)
    st.image(image, caption="📥 Uploaded Image", use_container_width=True)

    results = model.predict(np.array(image), conf=conf_threshold)
    res_plotted = results[0].plot()[:, :, ::-1]
    st.image(res_plotted, caption="🎯 Detection Result", use_container_width=True)

# -------------------------------
# 🎥 Video Detection
# -------------------------------
def detect_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
    cap.release()

# -------------------------------
# 🚀 Upload Section
# -------------------------------
uploaded_file = st.file_uploader("Upload your file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    if source_type == "Image" and uploaded_file.type.startswith("image"):
        detect_image(uploaded_file)
    elif source_type == "Video" and uploaded_file.type.startswith("video"):
        detect_video(uploaded_file)
    else:
        st.warning("⚠️ Please upload a valid file type according to your selection.")

st.markdown("---")
st.markdown("💡 **Tip:** Increase the confidence threshold for fewer but more accurate detections.")
