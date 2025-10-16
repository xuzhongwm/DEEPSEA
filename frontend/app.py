import streamlit as st
import torch
from PIL import Image
from classify_utils import *

st.set_page_config(page_title="app_page", layout="wide")

# --- Apply your gradient background ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,
            #EBE7DB 0%,
            #85ABB6 25%,
            #717A8B 50%,
            #5A7A9B 75%,
            #1C3659 100%) !important;
        background-attachment: fixed;
        color: #1C3659;
    }
    
    h1 {
    font-size: 70px !important;   /* main title */
    font-weight: 800 !important;
    color: #263238 !important;
    }
            
    p, div, span {
    font-size: 26px !important;   /* normal paragraph size */
    color: #263238 !important;
    }



    </style>
""", unsafe_allow_html=True)

st.title("🌊 Benthic Species Classification & Detection")
st.write("A computer vision app for benthic species detection and classification.")

st.write("This project brings computer vision underwater! We\'re " \
"using Google\'s Vision Transformer (ViT) and YOLO models to detect"
" and classify different benthic creatures—like eels, crabs, sponges, "
"and sea stars—from ocean images. YOLO helps find where each creature"
" is in the frame, while ViT figures out what species it is. " \
"Our goal is to make benthic research faster, more accurate, and a lot " \
"more fun by combining AI with ocean discovery.")

st.title("🖼️ ViT Image Classifier")

model, processor, load_msg = load_vit_model()
st.caption(load_msg)

tab1, tab2 = st.tabs(["Upload image", "Camera"])

with tab1:
    files = st.file_uploader("Drop images here", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if files:
        for f in files:
            image = Image.open(f).convert("RGB")
            st.image(image, caption=f.name, width=300)
            with st.spinner("Predicting..."):
                result = classify_image(image, model, processor)
            st.write(result)
            st.markdown("---")


with tab2:
    cam = st.camera_input("Take a photo")
    if cam is not None:
        image = Image.open(cam).convert("RGB")
        st.image(image, caption=cam.name, width=300)
        with st.spinner("Predicting..."):
            result = classify_image(image, model, processor)
        st.write(result)
        st.markdown("---")


