import os
import json
import numpy as np
import torch
from PIL import Image

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f"[ComfyUI-shaobkj] Error loading config.json: {e}")


def get_config_value(key, env_key, default):
    if env_key and os.environ.get(env_key):
        return os.environ.get(env_key)
    if key in CONFIG:
        return CONFIG[key]
    return default


def tensor_to_pil(image):
    t = image
    if isinstance(t, torch.Tensor) and t.dim() == 4:
        t = t[0]
    if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
        t = t.permute(1, 2, 0)
    arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
    pil = Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def pil_to_tensor(image):
    pil = image.convert("RGB") if hasattr(image, "convert") else image
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)


def resize_pil_long_side(image, long_side):
    try:
        target = int(long_side)
    except Exception:
        return image
    if target <= 0:
        return image
    w, h = image.size
    m = max(w, h)
    if m <= target:
        return image
    scale = target / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image
    return image.resize((new_w, new_h), resample=Image.LANCZOS)
