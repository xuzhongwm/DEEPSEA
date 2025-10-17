# DEEPSEA: Detection and Evaluation of Ecological Patterns in Submerged Environments using AI

Our project advances a next-generation, high-performance computer vision framework designed for accurate marine species classification and detection.

## Introduction

**DEEPSEA** is a next-generation computer vision toolkit designed for automated classification and detection of benthic marine species. Leveraging state-of-the-art deep learning frameworks such as **Google’s Vision Transformer (ViT)** for fine-grained species recognition and **YOLOv8** for real-time object detection, DEEPSEA bridges modern artificial intelligence with marine ecological research.

The toolkit enables researchers to accurately identify and localize underwater organisms—such as crabs, eels, and scallops—across large-scale ocean imagery datasets. By combining high-performance inference with an intuitive interface, **DEEPSEA** empowers marine biologists to accelerate biodiversity surveys, ecological monitoring, and conservation studies through intelligent, AI-driven analysis

### Getting started

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

### Highlights

**1. Model Performance**

Our models are trained on a curated benthic dataset containing thousands of labeled underwater images across seven species (e.g., crabs, scallops, eels, skates).

| Model                        | Task                   | Metric         | Accuracy  | Average Confidence |
| ---------------------------- | ---------------------- | -------------- | --------- | ------------------ |
| **Vision Transformer (ViT)** | Species Classification | Top-1 Accuracy | **≈ 92%** | **99.9%**          |
| **YOLOv8**                   | Object Detection       | mAP@0.5        | **≈ 85%** | **97–99%**         |

- **ViT (Classification):** Achieves ~92% accuracy with an average confidence of **99.9%**, indicating extremely strong model certainty across predictions.
- **YOLOv8 (Detection):** Delivers around 80% mean average precision (mAP) with **high confidence scores (97–99%)** for most detections, even under variable lighting and visibility.
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

**3. Real-Time Monitoring**

**DEEPSEA** is designed not only for post-collection image analysis but also for **real-time field monitoring**.

- **Live Image Capture:** Supports taking photos directly from connected devices (e.g., iPhone, laptop camera, or USB microscope).
- **Instant Inference:** Captured images are processed immediately through DEEPSEA’s integrated YOLOv8 and ViT pipelines for live detection and classification.
- **On-Site Monitoring:** Enables divers, researchers, and educators to analyze marine life directly in the field without complex software setup.

### Dive Deeper

[📄 Read the full YOLOv8 & ViT Overview (PDF)](https://github.com/Felikscjy/Benthic_detect/blob/main/DEEPSEA.pdf)

Our goal is to make benthic research faster, more accurate, and a lot more fun by combining AI with ocean discovery!
