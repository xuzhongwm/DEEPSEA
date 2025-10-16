import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import zipfile
from io import BytesIO
from datetime import datetime
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

def process_batch_images(images, model, processor, progress_bar=None):
    """批量处理图片并返回结果"""
    results = []
    
    for i, (filename, image) in enumerate(images):
        try:
            # 预测
            result = classify_image_with_probs(image, model, processor)
            result['filename'] = filename
            result['status'] = 'success'
            results.append(result)
            
            # 更新进度条
            if progress_bar:
                progress_bar.progress((i + 1) / len(images))
                
        except Exception as e:
            results.append({
                'filename': filename,
                'prediction': 'Error',
                'confidence': 0.0,
                'all_probs': {},
                'status': 'error',
                'error': str(e)
            })
    
    return results

def create_batch_report(results):
    """创建批量预测报告"""
    # 创建DataFrame
    df_data = []
    for result in results:
        if result['status'] == 'success':
            df_data.append({
                'Filename': result['filename'],
                'Prediction': result['prediction'],
                'Confidence': f"{result['confidence']:.2%}",
                'Status': 'Success'
            })
        else:
            df_data.append({
                'Filename': result['filename'],
                'Prediction': 'Error',
                'Confidence': 'N/A',
                'Status': 'Error'
            })
    
    df = pd.DataFrame(df_data)
    
    # 统计信息
    total_images = len(results)
    successful = len([r for r in results if r['status'] == 'success'])
    failed = total_images - successful
    
    # 预测分布
    if successful > 0:
        predictions = [r['prediction'] for r in results if r['status'] == 'success']
        pred_counts = pd.Series(predictions).value_counts()
    else:
        pred_counts = pd.Series()
    
    return df, total_images, successful, failed, pred_counts

def create_batch_summary_chart(pred_counts):
    """创建批量预测汇总图表"""
    if len(pred_counts) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pred_counts.index, pred_counts.values, color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Batch Prediction Summary', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
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

tab1, tab2, tab3 = st.tabs(["Single Image", "Camera", "Batch Upload"])

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

with tab3:
    st.header("📁 Batch Image Upload")
    st.write("Upload multiple images at once to get a comprehensive prediction report.")
    
    # 上传方式选择
    upload_method = st.radio(
        "Choose upload method:",
        ["Upload Folder (ZIP)", "Upload Multiple Files"],
        horizontal=True
    )
    
    uploaded_files = []
    
    if upload_method == "Upload Multiple Files":
        # 多文件上传
        uploaded_files = st.file_uploader(
            "Choose multiple images", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True,
            help="Select multiple images to process in batch"
        )
    
    else:
        # ZIP文件夹上传
        zip_file = st.file_uploader(
            "Upload a folder (ZIP file) containing images",
            type=["zip"],
            help="Compress your image folder into a ZIP file and upload it. All images in the folder will be processed automatically. (Ignores __MACOSX and hidden files)"
        )
        
        if zip_file is not None:
            try:
                # 提取ZIP文件
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # 获取ZIP中的所有文件
                    file_list = zip_ref.namelist()
                    
                    # 过滤掉__MACOSX文件夹和隐藏文件
                    filtered_files = []
                    for f in file_list:
                        # 忽略__MACOSX文件夹、隐藏文件和目录
                        if (not f.startswith('__MACOSX/') and 
                            not f.startswith('.') and 
                            not f.endswith('/') and
                            not f.startswith('Thumbs.db')):
                            filtered_files.append(f)
                    
                    # 过滤出图片文件
                    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
                    image_files = [f for f in filtered_files if any(f.lower().endswith(ext) for ext in image_extensions)]
                    
                    if image_files:
                        total_files = len(file_list)
                        filtered_count = len(filtered_files)
                        st.success(f"Found {len(image_files)} images in the ZIP file")
                        st.info(f"📊 Processed {total_files} total files, filtered to {filtered_count} valid files, found {len(image_files)} images")
                        
                        # 显示图片列表预览
                        with st.expander("📋 Preview images in folder"):
                            for i, img_file in enumerate(image_files[:10]):  # 只显示前10个
                                st.write(f"{i+1}. {img_file}")
                            if len(image_files) > 10:
                                st.write(f"... and {len(image_files) - 10} more images")
                        
                        # 创建虚拟文件对象
                        class VirtualFile:
                            def __init__(self, name, data):
                                self.name = name
                                self.data = data
                            
                            def read(self):
                                return self.data
                        
                        # 提取图片并创建虚拟文件对象
                        for img_file in image_files:
                            try:
                                img_data = zip_ref.read(img_file)
                                virtual_file = VirtualFile(img_file, img_data)
                                uploaded_files.append(virtual_file)
                            except Exception as e:
                                st.warning(f"Could not extract {img_file}: {str(e)}")
                    else:
                        st.error("No image files found in the ZIP file. Please ensure the ZIP contains PNG, JPG, or JPEG files.")
                        
            except Exception as e:
                st.error(f"Error processing ZIP file: {str(e)}")
                uploaded_files = []
    
    if uploaded_files:
        st.info(f"📊 **{len(uploaded_files)} images** selected for batch processing")
        
        # 处理选项
        col1, col2 = st.columns([1, 1])
        with col1:
            show_individual = st.checkbox("Show individual results", value=True)
        with col2:
            download_report = st.checkbox("Download report as CSV", value=True)
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            # 准备图片数据
            images = []
            for file in uploaded_files:
                try:
                    # 处理虚拟文件对象和普通文件对象
                    if hasattr(file, 'read'):
                        if hasattr(file, 'data'):  # 虚拟文件对象
                            image = Image.open(BytesIO(file.data)).convert("RGB")
                        else:  # 普通文件对象
                            image = Image.open(file).convert("RGB")
                    else:
                        st.error(f"Invalid file object: {file.name}")
                        continue
                    
                    images.append((file.name, image))
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
            
            if images:
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 批量处理
                status_text.text("🔄 Processing images...")
                results = process_batch_images(images, model, processor, progress_bar)
                
                # 生成报告
                status_text.text("📊 Generating report...")
                df, total, successful, failed, pred_counts = create_batch_report(results)
                
                # 显示结果
                st.success(f"✅ Batch processing completed! {successful}/{total} images processed successfully")
                
                # 统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images", total)
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                with col4:
                    success_rate = (successful / total * 100) if total > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # 预测分布图表
                if len(pred_counts) > 0:
                    st.subheader("📈 Prediction Distribution")
                    summary_chart = create_batch_summary_chart(pred_counts)
                    if summary_chart:
                        st.pyplot(summary_chart)
                    
                    # 详细种类统计
                    st.subheader("📊 Species Statistics")
                    species_stats = []
                    for species, count in pred_counts.items():
                        percentage = (count / successful * 100) if successful > 0 else 0
                        species_stats.append({
                            'Species': species,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    # 创建统计表格
                    stats_df = pd.DataFrame(species_stats)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # 显示统计摘要
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Most Common", pred_counts.index[0], f"{pred_counts.iloc[0]} images")
                    with col2:
                        if len(pred_counts) > 1:
                            st.metric("Second Most", pred_counts.index[1], f"{pred_counts.iloc[1]} images")
                        else:
                            st.metric("Second Most", "N/A", "0 images")
                    with col3:
                        diversity = len(pred_counts)
                        st.metric("Species Diversity", f"{diversity} species", f"out of 7 possible")
                
                # 详细结果表格
                st.subheader("📋 Detailed Results")
                st.dataframe(df, use_container_width=True)
                
                # 下载报告
                if download_report:
                    # 创建CSV
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"benthic_predictions_{timestamp}.csv"
                    
                    # 创建包含统计信息的完整报告
                    report_sections = []
                    
                    # 1. 总体统计
                    report_sections.append("=== BENTHIC SPECIES CLASSIFICATION REPORT ===")
                    report_sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    report_sections.append(f"Total Images: {total}")
                    report_sections.append(f"Successfully Processed: {successful}")
                    report_sections.append(f"Failed: {failed}")
                    report_sections.append(f"Success Rate: {success_rate:.1f}%")
                    report_sections.append("")
                    
                    # 2. 种类统计
                    if len(pred_counts) > 0:
                        report_sections.append("=== SPECIES STATISTICS ===")
                        for species, count in pred_counts.items():
                            percentage = (count / successful * 100) if successful > 0 else 0
                            report_sections.append(f"{species}: {count} images ({percentage:.1f}%)")
                        report_sections.append("")
                    
                    # 3. 详细结果
                    report_sections.append("=== DETAILED RESULTS ===")
                    report_sections.append(csv)
                    
                    # 合并报告
                    full_report = "\n".join(report_sections)
                    
                    st.download_button(
                        label="📥 Download Complete Report (CSV)",
                        data=full_report,
                        file_name=filename,
                        mime="text/csv"
                    )
                
                # 显示个别结果（可选）
                if show_individual:
                    st.subheader("🖼️ Individual Results")
                    for i, result in enumerate(results):
                        if result['status'] == 'success':
                            with st.expander(f"📸 {result['filename']} - {result['prediction']}"):
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    # 显示图片（需要重新加载）
                                    for file in uploaded_files:
                                        if file.name == result['filename']:
                                            try:
                                                if hasattr(file, 'data'):  # 虚拟文件对象
                                                    image = Image.open(BytesIO(file.data)).convert("RGB")
                                                else:  # 普通文件对象
                                                    image = Image.open(file).convert("RGB")
                                                st.image(image, width=200)
                                            except Exception as e:
                                                st.error(f"Could not display image: {str(e)}")
                                            break
                                with col2:
                                    st.success(f"**Prediction:** {result['prediction']}")
                                    st.info(f"**Confidence:** {result['confidence']:.2%}")
                                    
                                    # 显示softmax图
                                    if result['all_probs']:
                                        fig = create_softmax_chart(result['all_probs'], result['prediction'], result['confidence'])
                                        st.pyplot(fig)
                        else:
                            st.error(f"❌ {result['filename']}: {result.get('error', 'Unknown error')}")


