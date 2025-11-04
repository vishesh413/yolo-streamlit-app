import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image

st.title("🎥 Student Behavior Detection using YOLO")

# Load model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Upload option
option = st.radio("Select input type:", ["Image", "Video"])

if option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Detecting..."):
            results = model.predict(source=img, conf=0.25)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Result", use_column_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("Processing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("✅ Video processing completed!")
