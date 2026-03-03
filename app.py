import streamlit as st
from cvit_prediction import predict_video   # adjust if function name differs
import os
import urllib.request

MODEL_URL = "https://huggingface.co/mhamza-007/cvit_deepfake_detection/resolve/main/cvit2_deepfake_detection_ep_50.pth"
MODEL_PATH = "weight/cvit2_deepfake_detection_ep_50.pth"

# download model automatically
if not os.path.exists(MODEL_PATH):
    os.makedirs("weight", exist_ok=True)
    print("Downloading model... (first run only)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

st.title("Deepfake Video Detection System")

uploaded_file = st.file_uploader(
    "Upload a Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    st.write("Processing video...")

    result = predict_video(uploaded_file)

    st.success(f"Prediction: {result}")
