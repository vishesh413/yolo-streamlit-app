import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🤖", layout="wide")

# ---------- HEADER ----------
st.title("🤖 Real-Time Object Detection App")
st.markdown("### Powered by **YOLOv8 + Streamlit** 🚀")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
source_option = st.sidebar.radio("Select Input Type", ("📸 Upload Image", "🎥 Upload Video"))

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ---------- DETECTION FUNCTION ----------
def detect_objects(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    results = model(image_np, conf=confidence_threshold)
    annotated_frame = results[0].plot()
    return annotated_frame, results

# ---------- IMAGE UPLOAD ----------
if source_option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting objects... ⏳"):
                annotated_frame, results = detect_objects(image)
                st.image(annotated_frame, caption="✅ Detection Complete", use_container_width=True)

                # Optional: Show detections summary
                detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
                st.write("**Detected Objects:**", list(set(detected_classes)))

# ---------- VIDEO UPLOAD ----------
elif source_option == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)

        if st.button("🎯 Run Detection on Video"):
            with st.spinner("Processing video... ⏳"):
                output_path = "output.mp4"
                cap = cv2.VideoCapture(tfile.name)
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
                st.success("✅ Video Detection Complete!")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("👨‍💻 Developed by [Vishesh] 💫 | YOLOv8 + Streamlit App")
