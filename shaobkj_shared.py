import os
import json
import numpy as np
import torch
from PIL import Image
import time
import requests
from urllib.parse import urlparse

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


def disable_insecure_request_warnings():
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def create_requests_session(use_system_proxy: bool):
    session = requests.Session()
    session.trust_env = bool(use_system_proxy)
    if not use_system_proxy:
        session.proxies = {}
    proxies = {} if not use_system_proxy else None
    return session, proxies


def build_submit_timeout(wait_seconds: int):
    w = int(wait_seconds)
    # 如果用户设为0，给予极长的超时（24小时）以支持无限等待
    # 如果用户设为非0，给予至少30秒的缓冲，但尊重用户的设置，不再强制延长到600秒
    read_timeout = 86400 if w == 0 else max(30, w)
    return (10, read_timeout)


def post_json_with_retry(
    session,
    url,
    *,
    headers,
    payload,
    timeout,
    proxies,
    verify=False,
    max_retries=3,
):
    return post_with_retry(
        session,
        url,
        headers=headers,
        timeout=timeout,
        proxies=proxies,
        verify=verify,
        max_retries=max_retries,
        json=payload,
    )


def post_with_retry(
    session,
    url,
    *,
    headers,
    timeout,
    proxies,
    verify=False,
    max_retries=3,
    **request_kwargs,
):
    last_exc = None
    resp = None
    for attempt in range(1, int(max_retries) + 1):
        try:
            resp = session.post(
                url,
                headers=headers,
                timeout=timeout,
                verify=verify,
                proxies=proxies,
                **request_kwargs,
            )
        except requests.exceptions.ReadTimeout:
            raise
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            time.sleep(2 * attempt)
            continue

        if resp.status_code in (500, 502, 503, 504) and attempt < max_retries:
            time.sleep(2 * attempt)
            continue
        return resp
    if resp is not None:
        return resp
    raise RuntimeError(f"Connection Failed: {last_exc}")


def auth_headers_for_same_origin(url: str, api_origin: str, headers: dict):
    try:
        if not headers or not api_origin:
            return None
        if urlparse(str(url)).netloc != str(api_origin):
            return None
        return headers
    except Exception:
        return None
