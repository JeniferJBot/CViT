import streamlit as st
from cvit_prediction import predict_video   # adjust if function name differs

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
