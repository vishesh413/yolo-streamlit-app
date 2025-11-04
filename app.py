import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Detection App", layout="wide")
st.title("🎥 YOLOv8 Object Detection App")

# Model load karna
@st.cache_resource
def load_model():
    model_path = "best.pt"  # same folder me honi chahiye
    model = YOLO(model_path)
    return model

model = load_model()
st.sidebar.header("Choose Input Type")
option = st.sidebar.radio("Select type:", ["Image", "Video"])

# IMAGE DETECTION
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        st.image(image_path, caption="Uploaded Image", use_container_width=True)
        st.write("Detecting objects... ⏳")

        results = model.predict(source=image_path, conf=0.5)
        result_image = results[0].plot()

        st.image(result_image, caption="Detection Result", use_container_width=True)
        st.success("✅ Detection complete!")

# VIDEO DETECTION
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        st.video(video_path)
        st.write("Processing video... ⏳")

        cap = cv2.VideoCapture(video_path)
        output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        st.success("✅ Video processing complete!")
        st.video(output_path)

st.markdown("---")
st.caption("Made by Vishesh 🧠 using YOLOv8 + Streamlit")
