import streamlit as st

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Benthic Species Detection", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    /* Center and limit main content width */
    .block-container {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Prevent sidebar overlapping content */
    [data-testid="stSidebar"] {
        z-index: 1;
    }

    /* Set global font color */
    .stApp {
        color: black;
    }

    /* Ensure text elements stay black */
    .stMarkdown, .stText, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput {
        color: black !important;
    }

    /* Title color */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* Paragraph and label color */
    p, div, span, label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Page Title
# ----------------------------
st.markdown(
    """
    <div style='text-align: center; margin-top: 0.5em; margin-bottom: 1.5em;'>
        <h1 style='font-size: 48px; margin-bottom: 0.1em;'>
            DEEPSEA: Detection and Evaluation of Ecological Patterns in Submerged Environments using AI
        </h1>
        <p style='font-size: 20px; margin-top: 0; color: #333333;'>
            Our project advances a next-generation, high-performance computer vision framework 
            designed for accurate marine species classification and detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Main Content
# ----------------------------
st.markdown("""
## Introduction

**DEEPSEA** is a next-generation computer vision toolkit designed for automated classification and detection of benthic marine species. Leveraging state-of-the-art deep learning frameworks such as **Google’s Vision Transformer (ViT)** for fine-grained species recognition and **YOLOv8** for real-time object detection, DEEPSEA bridges modern artificial intelligence with marine ecological research.

The toolkit enables researchers to accurately identify and localize underwater organisms—such as crabs, eels, and scallops—across large-scale ocean imagery datasets. By combining high-performance inference with an intuitive interface, **DEEPSEA** empowers marine biologists to accelerate biodiversity surveys, ecological monitoring, and conservation studies through intelligent, AI-driven analysis.


### Getting Started

Visit the live DEEPSEA here:  
🔗 [https://unturgidly-unbesprinkled-alycia.ngrok-free.dev/](https://unturgidly-unbesprinkled-alycia.ngrok-free.dev/)

No installation is required — simply open the link in your browser to start exploring underwater AI.

1. **Navigate to the Detection or Classification Page**
   - Detection: Upload an image to locate benthic species (e.g., crabs, eels, scallops).
   - Classification: Upload an image to identify species with confidence scores.

2. **Upload Your Image(s)**
   - Supported formats: `.jpg`, `.png`, `.jpeg`
   - You can also upload multiple images or a ZIP file for batch processing.

3. **Run the Model**
   - The app automatically performs inference using pre-trained ViT and YOLO models.
   - Results appear with bounding boxes, confidence scores, and species labels.

4. **View and Analyze Results**
   - Interactive visualizations and summary statistics are displayed instantly.
   - Download processed outputs or view model performance metrics.
""")

# ----------------------------
# Highlights Section
# ----------------------------
st.markdown("""
### Highlights

**1. Model Performance**

Our models are trained on a curated benthic dataset containing thousands of labeled underwater images across seven species (e.g., crabs, scallops, eels, skates).

| Model | Task | Metric | Accuracy | Average Confidence |
|--------|-------|----------|-----------|--------------------|
| **Vision Transformer (ViT)** | Species Classification | Top-1 Accuracy | **≈ 92%** | **99.9%** |
| **YOLOv8** | Object Detection | mAP@0.5 | **≈ 85%** | **97–99%** |

- **ViT (Classification):** Achieves ~92% accuracy with an average confidence of **99.9%**, indicating extremely strong model certainty across predictions.  
- **YOLOv8 (Detection):** Delivers around 85% mean average precision (mAP) with **high confidence scores (97–99%)** for most detections, even under variable lighting and visibility.  
- Both models maintain **high-confidence predictions** while generalizing well across benthic categories, ensuring stable performance on unseen underwater imagery.

**2. Data Analysis**

**DEEPSEA** includes a built-in **data analysis module** that summarizes key statistics from batch image processing runs. After uploading multiple images or ZIP datasets, the system automatically aggregates performance and biodiversity metrics for easy interpretation. For each processed batch, DEEPSEA provides:
- **Species Distribution:** Counts and percentages of each detected species.  
- **Detection Confidence:** Mean and standard deviation of YOLO and ViT confidence scores.  
- **Spatial Metrics:** Average bounding box sizes, density of detections per image, and clustering across frames.  
- **Accuracy Overview:** Average classification accuracy, precision, and recall for the uploaded dataset.  
- **Processing Summary:** Total images analyzed, processing time, and throughput rate (images per second).

**Example Output**
| Metric | Example Value |
|--------|----------------|
| Images Processed | 250 |
| Avg. Confidence (ViT) | 99.8% |
| Avg. mAP (YOLOv8) | 85.3% |
| Dominant Species | Crab (42%) |
| Avg. Processing Time | 0.8s/image |
""")

# ----------------------------
# Two Images Side-by-Side
# ----------------------------
st.markdown(
    """
    <p align="center" style="display: flex; justify-content: center; align-items: flex-start;">
      <img src="https://raw.githubusercontent.com/Felikscjy/Benthic_detect/main/frontend/images/curve.png" 
           alt="Curve Image" width="45%" style="margin-right:10px;"/>
      <img src="https://raw.githubusercontent.com/Felikscjy/Benthic_detect/main/frontend/images/normalized.png" 
           alt="Normalized Curve Image" width="45%; margin-top:15px;"/>
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Real-Time Monitoring Section
# ----------------------------
st.markdown("""
**3. Real-Time Monitoring**

**DEEPSEA** is designed not only for post-collection image analysis but also for **real-time field monitoring**.

- **Live Image Capture:** Supports taking photos directly from connected devices (e.g., iPhone, laptop camera, or USB microscope).  
- **Instant Inference:** Captured images are processed immediately through DEEPSEA’s integrated YOLOv8 and ViT pipelines for live detection and classification.  
- **On-Site Monitoring:** Enables divers, researchers, and educators to analyze marine life directly in the field without complex software setup.  


### Dive Deeper
[📄 Read the full YOLOv8 & ViT Overview (PDF)](https://github.com/Felikscjy/Benthic_detect/blob/main/DEEPSEA.pdf)

Our goal is to make benthic research faster, more accurate, and a lot more fun by combining AI with ocean discovery!
""")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:14px;'>"
    "Built with ❤️ for marine biology research"
    "</p>",
    unsafe_allow_html=True
)
