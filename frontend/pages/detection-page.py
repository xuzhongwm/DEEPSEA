import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import tempfile
import os
import io
import pandas as pd
from datetime import datetime

st.markdown(
    """
    <style>
    /* horizontally centered */
    .block-container {
        max-width: 1100px;        /* max width */
        margin: 0 auto;           /* auto center */
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* prevent sidebar squeezing the content */
    [data-testid="stSidebar"] {
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set wide layout
st.set_page_config(page_title="Benthic Species Detection", layout="wide")

# Page title
st.markdown(
    "<h1 style='text-align: center; font-size: 40px;'>🐋 YOLOv8 Benthic Species Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray; font-size:16px;'>"
    "★ Upload an image or batch of images to detect benthic species using YOLOv8. ★"
    "</p>",
    unsafe_allow_html=True
)

# load model with caching
@st.cache_resource
def load_model():
    model = YOLO("../best.pt")
    return model

model = load_model()

#initialize session state
if 'clear_uploader' not in st.session_state:
    st.session_state.clear_uploader = 0

tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Upload", "Camera"])

with tab1:
    # file uploader
    with st.container():
        st.markdown(
            """
            <h4 style='text-align: left; font-weight: 600; font-size: 16px; margin-top: 0px;'>
                Upload a Single Image for Detection:
            </h4>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="single_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # show original image first
        empty_col1, col1, col2, empty_col2 = st.columns([.1, 3, 3, .1])
        with col1:
                st.image(image, caption="Original Image", use_container_width=False, width=450)
                # Set run detection button
                btn_center = st.columns([4, 4, 4])
                with btn_center[1]:
                    run = st.button("Run Detection", use_container_width=True)

        # if detection started
        if run:
            results = model.predict(image)
            result_image = results[0].plot()
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                st.info("No species detected!")
            else: 
                # show detection result image
                with col2:
                    st.image(result_image, caption="Detection Result", use_container_width=False, width=450)

with tab2:
    with st.container():
        st.markdown(
            """
            <h4 style='text-align: left; font-weight: 600; font-size: 16px; margin-top: 0px;'>
                Upload a Batch of Images for Detection:
            </h4>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_files = st.file_uploader(
            "label", 
            type=["jpg","jpeg","png","zip"], 
            accept_multiple_files=True, 
            label_visibility="collapsed",
            key=f"batch_upload_{st.session_state.clear_uploader}"
        )

    images_list = []

    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                try:
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        for f in zip_ref.namelist():
                            if f.lower().endswith((".jpg",".jpeg",".png")) and not f.startswith("__MACOSX/"):
                                img_bytes = zip_ref.read(f)
                                img = Image.open(io.BytesIO(img_bytes))
                                images_list.append((f, img))
                except Exception as e:
                    st.error(f"Error reading ZIP file {file.name}: {str(e)}")
            else:
                img = Image.open(file)
                images_list.append((file.name, img))

        if images_list:
            st.markdown("#### Uploaded Images Preview")
            st.image([img for _, img in images_list], width=150, caption=[name for name,_ in images_list])

            col1, col2 = st.columns([10, 2])
            with col2:
                if st.button("Remove All", type="primary"):
                    st.session_state.clear_uploader += 1
                    st.rerun()

            run_batch = st.button("Run Detection on All Images", key="batch_run")

            if run_batch:
                total_counts = Counter()
                detailed_records = []

                with st.expander("Show All Detection Results", expanded=False):
                    for idx, (name, img) in enumerate(images_list):
                        results = model.predict(img)
                        labels = [int(cls) for cls in results[0].boxes.cls]
                        species_names = [model.names[i] for i in labels]
                        total_counts.update(labels)

                        detected_str = ", ".join(species_names) if species_names else "None"
                        detailed_records.append({
                            "Image Name": name,
                            "Detected Species": detected_str,
                            "Count": len(species_names)
                        })

                        # Only show image if there are detections!
                        if len(labels) > 0:
                            with st.expander(f"Detection result for {name}", expanded=(idx == 0)):
                                result_image = results[0].plot()
                                st.image(result_image, caption=f"Detected: {name}", width=400)
                        else:
                            with st.expander(f"No detection in {name}", expanded=False):
                                st.info("No benthic species detected in this image.")


                st.markdown("---")
                st.markdown(
                    "<h1 style='text-align: center; font-size: 30px;'>💡 Batch Detection Summary</h1>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<p style='text-align:center; color:gray; font-size:16px;'>"
                    "★ Here is a statistic report for all benthic image data you uploaded. ★"
                    "</p>",
                    unsafe_allow_html=True
                )

                # --- summary section ---
                total = len(images_list)
                detected = len([r for r in detailed_records if r["Count"] > 0])
                not_detected = total - detected
                detection_rate = (detected / total * 100) if total > 0 else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images", total)
                with col2:
                    st.metric("Detected", detected)
                with col3:
                    st.metric("No Benthic Species Detected", not_detected)
                with col4:
                    st.metric("Detection Coverage", f"{detection_rate:.1f}%")

                # --- charts ---
                if total_counts:
                    categories = [model.names[i] for i in total_counts.keys()]
                    values = list(total_counts.values())

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), gridspec_kw={'wspace':0.6})
                    colors = plt.get_cmap("tab20").colors[:len(values)]

                    # Pie chart
                    wedges, _, _ = ax1.pie(
                        values, autopct="%1.1f%%", startangle=90,
                        counterclock=False, colors=colors
                    )
                    ax1.set_title("Cumulative Species Distribution (Pie)")
                    ax1.axis('equal')

                    # Bar chart
                    ax2.bar(categories, values, color=colors)
                    ax2.set_title("Cumulative Species Counts (Bar)")
                    ax2.set_ylabel("Count")
                    ax2.set_xticklabels(categories, rotation=45, ha='right')
                    st.pyplot(fig, clear_figure=True, use_container_width=True)

                    # --- Species table ---
                    st.subheader("Species Statistics")
                    species_stats = []
                    for species, count in total_counts.items():
                        percentage = (count / sum(values) * 100) if sum(values) > 0 else 0
                        species_stats.append({
                            "Species": model.names[species],
                            "Count": count,
                            "Percentage": f"{percentage:.1f}%"
                        })
                    stats_df = pd.DataFrame(species_stats)
                    st.dataframe(stats_df, use_container_width=True)

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    sorted_species = stats_df.sort_values(by="Count", ascending=False)
                    with col1:
                        st.metric("Most Common", sorted_species.iloc[0]["Species"], f"{sorted_species.iloc[0]['Count']} images")
                    with col2:
                        if len(sorted_species) > 1:
                            st.metric("Second Most", sorted_species.iloc[1]["Species"], f"{sorted_species.iloc[1]['Count']} images")
                        else:
                            st.metric("Second Most", "N/A", "0 images")
                    with col3:
                        st.metric("Species Diversity", len(stats_df), "species detected")

                # --- Detailed results table ---
                st.subheader("Detailed Results")
                df = pd.DataFrame(detailed_records)
                st.dataframe(df, use_container_width=True)

                # --- Download report ---
                csv = df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"benthic_batch_report_{timestamp}.csv"

                st.download_button(
                    label="Download Detailed Report (CSV)",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )

with tab3:
    st.markdown(
        """
        <h4 style='text-align: left; font-weight: 600; font-size: 16px; margin-top: 0px;'>
            Take a Photo for Detection:
        </h4>
        """,
        unsafe_allow_html=True
    )

    cam_file = st.camera_input("", label_visibility="collapsed")

    if cam_file is not None:
        image = Image.open(cam_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Camera Photo", width=300)

        with col2:
            run_cam = st.button("Run Detection", key="camera_run")

            if run_cam:
                with st.spinner("Detecting..."):
                    results = model.predict(image)
                    boxes = results[0].boxes

                    if boxes is None or len(boxes) == 0:
                        st.info("No benthic species detected!")
                    else:
                        result_image = results[0].plot()
                        st.image(result_image, caption="Detection Result", width=300)

                        labels = [int(cls) for cls in results[0].boxes.cls]
                        species_names = [model.names[i] for i in labels]
                        counts = Counter(species_names)

                        st.subheader("Detection Summary")
                        for species, count in counts.items():
                            st.write(f"{species}: {count}")