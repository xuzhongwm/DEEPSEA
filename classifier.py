import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

model_dir = "./vit-single-out/best"
processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)  # quiets the warning
model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
model.eval()

img = Image.open("./example.png").convert("RGB")

with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # <<< move inputs to SAME device
    logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]

id2label = model.config.id2label
topk_probs, topk_ids = torch.topk(probs, k=5)
print({id2label[int(i)]: float(p) for p, i in zip(topk_probs, topk_ids)})
print(f"The image is {id2label[topk_ids[0].item()]}")


