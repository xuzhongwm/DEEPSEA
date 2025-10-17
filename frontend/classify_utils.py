import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F

def load_vit_model(model_dir="../vit-single-out/vit-single-out/best", device="cpu"):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor, "loading model..."

def classify_image_softmax(image: Image.Image, model, processor, device="cpu", topk=5):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        topk_probs, topk_ids = torch.topk(probs, k=topk)
        id2label = model.config.id2label
        return [
            (id2label[int(i)], float(p))
            for p, i in zip(topk_probs, topk_ids)
        ]

def classify_image(image: Image.Image, model, processor, device="cpu", topk=5):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        topk_probs, topk_ids = torch.topk(probs, k=topk)
        id2label = model.config.id2label
        return f"Result: {id2label[topk_ids[0].item()]}"

def classify_image_with_probs(image: Image.Image, model, processor, device="cpu"):
    """返回预测结果和所有类别的概率"""
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(logits.argmax(-1))
        id2label = model.config.id2label
        
        # 返回预测结果和所有概率
        return {
            'prediction': id2label[pred_id],
            'confidence': float(probs[pred_id]),
            'all_probs': {id2label[i]: float(probs[i]) for i in range(len(probs))}
        }