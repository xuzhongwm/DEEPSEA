import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch.nn.functional as F

def load_vit_model(model_dir="../vit-single-out/best", device="cpu"):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, processor, "loading model..."

def classify_image_softmax(image: Image.Image, model, processor, device="cpu", topk=5):
    inputs = processor(images=image, return_tensors="pt").to(device)
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
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        topk_probs, topk_ids = torch.topk(probs, k=topk)
        id2label = model.config.id2label
        return f"Result: {id2label[topk_ids[0].item()]}"