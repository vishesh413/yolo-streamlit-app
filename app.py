import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image

# -----------------------------------------------------------
# 🌈 Custom CSS Styling
# -----------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    h1 {
        text-align: center;
        color: #fff;
        text-shadow: 2px 2px 6px #00000088;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7575;
        transform: scale(1.05);
    }
    .stRadio label {
        color: #fff !important;
        font-size: 18px;
        font-weight: 600;
    }
    .stFileUploader label {
        color: #fff !important;
        font-size: 17px;
    }
    .css-1y4p8pa {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# 🧠 Title and Description
# -----------------------------------------------------------
st.title("🎥 Student Behavior Detection using YOLO")
st.markdown("""
Welcome to the **YOLO-based Student Behavior Detection System** 👨‍🏫  
Upload an **image** or **video** below and watch how the model detects student actions in real time.
""")

# -----------------------------------------------------------
# 🚀 Load YOLO Model
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# -----------------------------------------------------------
# 🎛️ Choose Input Type
# -----------------------------------------------------------
option = st.radio("📁 Select Input Type:", ["Image", "Video"])

# -----------------------------------------------------------
# 🖼️ IMAGE Detection
# -----------------------------------------------------------
if option == "Image":
    uploaded_image = st.file_uploader("📸 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Detecting... Please wait"):
            results = model.predict(source=img, conf=0.25)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="✅ Detection Result", use_column_width=True)

# -----------------------------------------------------------
# 🎥 VIDEO Detection
# -----------------------------------------------------------
elif option == "Video":
    uploaded_video = st.file_uploader("🎬 Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("⚙️ Processing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("🎉 Video processing completed successfully!")

# -----------------------------------------------------------
# 👣 Footer
# -----------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:#fff;'>Developed with ❤️ using Streamlit & YOLO</p>", unsafe_allow_html=True)
