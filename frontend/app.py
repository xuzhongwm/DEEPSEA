import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from classify_utils import *

st.set_page_config(page_title="app_page", layout="wide")

def create_softmax_chart(probs_dict, prediction, confidence):
    """创建softmax概率柱状图"""
    # 按概率排序
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    
    # 创建颜色映射，预测结果用特殊颜色
    colors = ['#FF6B6B' if label == prediction else '#4ECDC4' for label in labels]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # 设置标签
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Classification Results\nPredicted: {prediction} (Confidence: {confidence:.2%})', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴标签
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # 在柱子上添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 设置y轴范围
    ax.set_ylim(0, max(values) * 1.2)
    
    # 添加网格
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置背景色
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    return fig

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

    .stImage > div {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 15px;
        overflow: hidden;
    }
    
    .stImage > div:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* 按钮悬停效果 */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 25px;
        background: linear-gradient(45deg, #1C3659, #5A7A9B);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(28, 54, 89, 0.4);
    }
    
    /* 文件上传区域悬停效果 */
    .stFileUploader > div {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stFileUploader > div:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(28, 54, 89, 0.2);
    }

    /* 标签页悬停效果 */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 2px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(45deg, rgba(28, 54, 89, 0.8), rgba(90, 122, 155, 0.8));
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(28, 54, 89, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1C3659, #5A7A9B);
        color: white !important;
        box-shadow: 0 2px 8px rgba(28, 54, 89, 0.4);
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
            
            # 创建两列布局
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption=f.name, width=300)
            
            with col2:
                with st.spinner("Predicting..."):
                    # 获取详细预测结果
                    result = classify_image_with_probs(image, model, processor)
                
                # 显示预测结果
                st.success(f"🎯 **Prediction:** {result['prediction']}")
                st.info(f"📊 **Confidence:** {result['confidence']:.2%}")
                
                # 创建并显示softmax图
                fig = create_softmax_chart(result['all_probs'], result['prediction'], result['confidence'])
                st.pyplot(fig)
            
            st.markdown("---")


with tab2:
    cam = st.camera_input("Take a photo")
    if cam is not None:
        image = Image.open(cam).convert("RGB")
        
        # 创建两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Camera Photo", width=300)
        
        with col2:
            with st.spinner("Predicting..."):
                # 获取详细预测结果
                result = classify_image_with_probs(image, model, processor)
            
            # 显示预测结果
            st.success(f"🎯 **Prediction:** {result['prediction']}")
            st.info(f"📊 **Confidence:** {result['confidence']:.2%}")
            
            # 创建并显示softmax图
            fig = create_softmax_chart(result['all_probs'], result['prediction'], result['confidence'])
            st.pyplot(fig)
        
        st.markdown("---")


