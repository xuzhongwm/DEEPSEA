import streamlit as st

st.set_page_config(page_title="Benthic Species Detection", layout="wide")

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
    
    /* 设置文字颜色为白色 */
    .stApp {
        color: black;
    }
    
    /* 设置主要文字颜色为白色 */
    .stMarkdown, .stText, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput {
        color: black !important;
    }
    
    /* 设置标题颜色为白色 */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    
    /* 设置段落和文本颜色为白色 */
    p, div, span, label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.markdown(
    "<h1 style='text-align: center; font-size: 50px;'>🌊 Benthic Species Detection & Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:gray; font-size:18px;'>"
    "★ Welcome to our AI-powered marine biology research platform! ★"
    "</p>",
    unsafe_allow_html=True
)

# Main content
st.markdown("""
## 🚀 About This Project

This project brings computer vision underwater! We're using cutting-edge AI models to detect and classify different benthic creatures—like eels, crabs, sponges, and sea stars—from ocean images.

### 🔬 Our Technology Stack

- **YOLOv8**: For object detection and localization
- **Vision Transformer (ViT)**: For species classification
- **Streamlit**: For interactive web interface
- **PyTorch**: For deep learning inference

### 🎯 What We Can Do

1. **Detection**: Find where each creature is located in the image
2. **Classification**: Identify what species each creature belongs to
3. **Batch Processing**: Analyze multiple images at once
4. **Statistical Reports**: Generate comprehensive analysis reports

### 🌊 Supported Species

Our models can identify 7 different benthic species:
- **Eel**
- **Scallop**
- **Crab**
- **Flatfish**
- **Roundfish**
- **Skate**
- **Whelk**

### 🎮 How to Use

1. **Detection Page**: Upload images to detect and locate benthic species
2. **Classification Page**: Upload images to classify species with confidence scores
3. **Batch Processing**: Upload multiple images or ZIP folders for comprehensive analysis

Our goal is to make benthic research faster, more accurate, and a lot more fun by combining AI with ocean discovery!

---

### 📊 Model Performance

- **Detection Accuracy**: High precision in locating marine creatures
- **Classification Accuracy**: 92%+ accuracy on validation set
- **Processing Speed**: Real-time inference capabilities
- **Batch Processing**: Handle hundreds of images efficiently

### 🔬 Research Applications

- Marine biodiversity surveys
- Environmental monitoring
- Species population studies
- Educational research tools
- Conservation efforts

---

**Ready to explore the underwater world? Navigate to the Detection or Classification pages to get started!**
""")

# Add some visual elements
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Detection
    Find and locate benthic species in images using YOLOv8
    """)

with col2:
    st.markdown("""
    ### 🧠 Classification
    Identify species with confidence scores using ViT
    """)

with col3:
    st.markdown("""
    ### 📊 Analysis
    Generate comprehensive reports and statistics
    """)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:14px;'>"
    "Built with ❤️ for marine biology research"
    "</p>",
    unsafe_allow_html=True
)
