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

# initialize session state
if 'clear_uploader' not in st.session_state:
    st.session_state.clear_uploader = 0

tab1, tab2 = st.tabs(["Single Image", "Batch Upload"])

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
                st.info("No species detected to plot!")
            else: 
                # show detection result image
                with col2:
                    st.image(result_image, caption="Detection Result", use_container_width=False, width=450)

                # # Pie chart of detected species
                # labels = [int(cls) for cls in results[0].boxes.cls]
                # counts = Counter(labels)
                # categories = [model.names[i] for i in counts.keys()]
                # values = list(counts.values())

                # if values:
                #     st.markdown("---")
                #     fig, ax = plt.subplots(figsize=(7,7))
                #     colors = plt.get_cmap("tab20").colors[:len(values)]

                #     wedges, texts, autotexts = ax.pie(
                #         values,
                #         labels=None,
                #         autopct="%1.1f%%",
                #         pctdistance=0.7,
                #         startangle=90,
                #         counterclock=False,
                #         wedgeprops={"linewidth":1, "edgecolor":"white"},
                #         colors=colors
                #     )

                #     # add labels
                #     for i, p in enumerate(wedges):
                #         if values[i] < 0.01:  
                #             continue
                #         ang = (p.theta2 - p.theta1)/2. + p.theta1
                #         y = np.sin(np.deg2rad(ang))
                #         x = np.cos(np.deg2rad(ang))
                #         ha = {-1: "right", 1: "left"}[int(np.sign(x))]
                #         ax.annotate(
                #             f"{categories[i]}",
                #             xy=(x, y),
                #             xytext=(1.3*np.sign(x), 1.3*y),
                #             horizontalalignment=ha,
                #             arrowprops=dict(arrowstyle="-"),
                #             fontsize=10
                #         )

                #     ax.set_title("Detected Species Distribution", fontsize=12)
                #     ax.axis('equal')
                #     st.pyplot(fig, clear_figure=True, use_container_width=False, width=400)
                # else:
                #     st.info("No species detected to plot!")

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
        
        # use key to reset uploader
        uploaded_files = st.file_uploader(
            "", 
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

        # image preview
        if images_list:
            st.markdown("#### Uploaded Images Preview")
            st.image([img for _, img in images_list], width=150, caption=[name for name,_ in images_list])

            # Remove All button
            col1, col2 = st.columns([10, 2])
            with col2:
                if st.button("Remove All", type="primary"):
                    st.session_state.clear_uploader += 1
                    st.rerun()

            run_batch = st.button("Run Detection on All Images", key="batch_run")
            if run_batch:
                total_counts = Counter()

                # larger expander for all results
                with st.expander("Show All Detection Results", expanded=False):
                    for idx, (name, img) in enumerate(images_list):
                        # each image result in its own expander
                        with st.expander(f"Detection result for {name}", expanded=(idx == 0)):
                            results = model.predict(img)
                            result_image = results[0].plot()
                            st.image(result_image, caption=f"Detected: {name}", width=400)

                            # add up total counts in batch
                            labels = [int(cls) for cls in results[0].boxes.cls]
                            total_counts.update(labels)

                
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
                st.write("\n\n\n")  
                
                if total_counts:
                    categories = [model.names[i] for i in total_counts.keys()]
                    values = list(total_counts.values())

                    # two plots side by side
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), gridspec_kw={'wspace':0.6})

                    # --- pie chart ---
                    colors = plt.get_cmap("tab20").colors[:len(values)]
                    wedges, texts, autotexts = ax1.pie(
                        values,
                        labels=None,
                        autopct="%1.1f%%",
                        pctdistance=0.7,
                        startangle=90,
                        counterclock=False,
                        wedgeprops={"linewidth":1, "edgecolor":"white"},
                        colors=colors
                    )
                    
                    for i, p in enumerate(wedges):
                        if values[i] < 0.01:
                            continue
                        ang = (p.theta2 - p.theta1)/2. + p.theta1
                        y = np.sin(np.deg2rad(ang))
                        x = np.cos(np.deg2rad(ang))
                        ha = {-1: "right", 1: "left"}[int(np.sign(x))]
                        ax1.annotate(
                            f"{categories[i]}",
                            xy=(x, y),
                            xytext=(1.3*np.sign(x), 1.3*y),
                            horizontalalignment=ha,
                            arrowprops=dict(arrowstyle="-"),
                            fontsize=10
                        )
                    ax1.set_title("Cumulative Species Distribution (Pie)", fontsize=12, pad=20)
                    ax1.axis('equal')

                    # --- bar chart ---
                    ax2.bar(categories, values, color=colors)
                    ax2.set_title("Cumulative Species Counts (Bar)", fontsize=12, pad=20)
                    ax2.set_ylabel("Count")
                    ax2.set_xticklabels(categories, rotation=45, ha='right')

                    st.pyplot(fig, clear_figure=True, use_container_width=True)
                else:
                    st.info("No species detected in any of the uploaded images.")