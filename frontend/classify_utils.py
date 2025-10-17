from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

REQUIRED_FILES = ("config.json", "preprocessor_config.json")

def _has_required_files(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    has_cfg = all((p / f).exists() for f in REQUIRED_FILES)
    has_weights = (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
    return has_cfg and has_weights

def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    """
    Try several likely locations (relative to this file and CWD).
    This avoids the “Model dir not found” when running Streamlit from different roots.
    """
    here = Path(__file__).resolve().parent          # .../frontend
    root = here.parent                               # repo root
    cwd  = Path.cwd()

    candidates: list[Path] = []
    # 1) If caller provided a path, try it relative to this file and CWD
    if model_dir:
        candidates += [(here / model_dir).resolve(), (cwd / model_dir).resolve()]

    # 2) Common layouts for your project (based on your screenshot)
    candidates += [
        # when the model lives at repo_root/vit-single-out/vit-single-out/best
        (root / "vit-single-out" / "vit-single-out" / "best").resolve(),
        # when the model was moved under frontend/
        (here / "vit-single-out" / "vit-single-out" / "best").resolve(),
        # simpler variant if you collapse duplicates later
        (root / "vit-single-out" / "best").resolve(),
        (here / "vit-single-out" / "best").resolve(),
    ]

    for p in candidates:
        if _has_required_files(p):
            return p

    # If none matched, raise with a helpful message showing what we checked
    checked = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate a valid model directory containing "
        "`config.json`, `preprocessor_config.json`, and weights.\n"
        f"Checked:\n  - {checked}"
    )

def load_vit_model(model_dir: str | Path | None = None, device: str = "cpu"):
    """
    Load locally-saved ViT fine-tune. We resolve the directory robustly and
    force local-only to avoid HF repo-id validation.
    """
    model_path = _resolve_model_dir(model_dir)

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    model.to(device)
    return model, processor, f"Loaded model from {model_path}"

def _to_device(batch, device: str):
    return {k: v.to(device) for k, v in batch.items()}

def classify_image_softmax(image: Image.Image, model, processor, device: str = "cpu", topk: int = 5):
    inputs = processor(images=image, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        topk_probs, topk_ids = torch.topk(probs, k=topk)
        id2label = model.config.id2label
        return [(id2label[int(i)], float(p)) for p, i in zip(topk_probs, topk_ids)]

def classify_image(image: Image.Image, model, processor, device: str = "cpu", topk: int = 5):
    inputs = processor(images=image, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        topk_probs, topk_ids = torch.topk(probs, k=topk)
        id2label = model.config.id2label
        return f"Result: {id2label[topk_ids[0].item()]}"

def classify_image_with_probs(image: Image.Image, model, processor, device: str = "cpu"):
    inputs = processor(images=image, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(logits.argmax(-1).item())
        id2label = model.config.id2label
        return {
            "prediction": id2label[pred_id],
            "confidence": float(probs[pred_id]),
            "all_probs": {id2label[i]: float(probs[i]) for i in range(len(probs))}
        }
