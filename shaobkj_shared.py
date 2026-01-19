import os
import json
import re
import shutil
import numpy as np
import torch
from PIL import Image
import time
import requests
import io
import base64
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
    if w == 0:
        return (10, 86400)
    if w < 0:
        w = 1
    connect_timeout = min(10, w)
    # Ensure read_timeout uses the user-provided wait_seconds (w)
    # The read_timeout should be at least as long as the user wants to wait.
    read_timeout = max(1, w)
    return (connect_timeout, read_timeout)


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
    
    # Extract total timeout limit if timeout is a tuple
    total_timeout_limit = timeout[1] if isinstance(timeout, (tuple, list)) and len(timeout) > 1 else timeout
    start_time = time.time()

    for attempt in range(1, int(max_retries) + 1):
        # Check total timeout before next attempt
        if total_timeout_limit is not None:
            elapsed = time.time() - start_time
            if elapsed >= total_timeout_limit:
                raise requests.exceptions.ReadTimeout(f"Total execution time exceeded limit of {total_timeout_limit}s")
            
            # Adjust request timeout for this attempt to respect total limit
            # If timeout is tuple (connect, read), adjust read part
            current_remaining = total_timeout_limit - elapsed
            if isinstance(timeout, (tuple, list)):
                connect_timeout = timeout[0]
                # Ensure we don't pass negative timeout
                current_attempt_timeout = (connect_timeout, max(0.1, current_remaining))
            else:
                current_attempt_timeout = max(0.1, current_remaining)
        else:
            current_attempt_timeout = timeout

        try:
            resp = session.post(
                url,
                headers=headers,
                timeout=current_attempt_timeout,
                verify=verify,
                proxies=proxies,
                **request_kwargs,
            )
        except requests.exceptions.ReadTimeout:
            # If it's the last attempt or total time exceeded, raise
            if attempt >= max_retries or (total_timeout_limit and (time.time() - start_time) >= total_timeout_limit):
                raise
            last_exc = requests.exceptions.ReadTimeout("Read timed out")
            # Don't sleep if we are close to timeout? 
            # Standard logic below
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            # Check if sleep would exceed total timeout
            sleep_time = 2 * attempt
            if total_timeout_limit and (time.time() - start_time + sleep_time) > total_timeout_limit:
                 # If sleeping would timeout, just raise or sleep less?
                 # Let's just raise last exception if we can't retry effectively
                 raise last_exc
            time.sleep(sleep_time)
            continue

        if resp.status_code in (500, 502, 503, 504) and attempt < max_retries:
            sleep_time = 2 * attempt
            if total_timeout_limit and (time.time() - start_time + sleep_time) > total_timeout_limit:
                 return resp # Return what we have if we can't retry
            time.sleep(sleep_time)
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


def extract_task_id_and_video_url(resp):
    resp_json = None
    resp_text = None
    try:
        resp_json = resp.json()
    except Exception:
        try:
            resp_text = resp.text
        except Exception:
            resp_text = None

    task_id = None
    video_url = None

    def maybe_set_from_dict(d):
        nonlocal task_id, video_url
        if not isinstance(d, dict):
            return
        if task_id is None:
            task_id = d.get("task_id") or d.get("id")
        if video_url is None:
            u = d.get("video_url") or d.get("url")
            if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
                video_url = u
        data = d.get("data")
        if isinstance(data, dict):
            if task_id is None:
                task_id = data.get("task_id") or data.get("id")
            if video_url is None:
                u = data.get("video_url") or data.get("url")
                if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
                    video_url = u
        result = d.get("result")
        if isinstance(result, dict):
            if task_id is None:
                task_id = result.get("task_id") or result.get("id")
            if video_url is None:
                u = result.get("video_url") or result.get("url")
                if isinstance(u, str) and (u.startswith("http://") or u.startswith("https://")):
                    video_url = u
        choices = d.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str) and content:
                return content
        return None

    content_text = None
    if isinstance(resp_json, dict):
        content_text = maybe_set_from_dict(resp_json)
        if resp_text is None and content_text:
            resp_text = content_text

    if not task_id:
        headers = getattr(resp, "headers", {}) or {}
        task_id = headers.get("X-Task-Id") or headers.get("Task-Id") or headers.get("task_id") or headers.get("task-id")
        if not task_id:
            location = headers.get("Location") or headers.get("location")
            if isinstance(location, str) and location:
                m = re.search(r"/([^/]+)/?$", location.strip())
                if m:
                    task_id = m.group(1)

    if video_url is None and isinstance(resp_text, str) and resp_text.strip():
        m = re.search(r"(https?://[^\s\)\]]+)", resp_text)
        if m:
            u = m.group(1).rstrip(".,;!?")
            if u.startswith("http://") or u.startswith("https://"):
                video_url = u

    return task_id, video_url, resp_json, resp_text


class SimpleVideoAdapter:
    def __init__(self, video_path_or_url: str):
        v = str(video_path_or_url or "")
        self.is_url = v.startswith("http://") or v.startswith("https://")
        self.video_url = v if self.is_url else None
        self.video_path = v if (not self.is_url and v) else None

    def get_dimensions(self):
        return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        try:
            if self.is_url and self.video_url:
                resp = requests.get(self.video_url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True
            if self.video_path and os.path.exists(self.video_path):
                shutil.copyfile(self.video_path, output_path)
                return True
            return False
        except Exception:
            return False

def extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60):
    """
    Extracts a PIL Image from an API response JSON.
    Supports Gemini (inlineData), OpenAI (b64_json, url), and markdown/raw URLs.
    Returns: PIL.Image object or None
    """
    if not isinstance(res_json, dict):
        return None

    # 1. Gemini / Google format
    if "candidates" in res_json and isinstance(res_json["candidates"], list) and res_json["candidates"]:
        for cand in res_json["candidates"]:
            content = cand.get("content") if isinstance(cand, dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    try:
                        image_data = base64.b64decode(inline["data"])
                        image = Image.open(io.BytesIO(image_data))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        return image
                    except Exception:
                        pass

    # 2. OpenAI / Generic format (data list)
    if "data" in res_json and isinstance(res_json["data"], list) and res_json["data"]:
        data_item = res_json["data"][0]
        if isinstance(data_item, dict):
            # 2a. b64_json
            if "b64_json" in data_item:
                try:
                    image_data = base64.b64decode(data_item["b64_json"])
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
                except Exception:
                    pass
            # 2b. url
            if "url" in data_item:
                try:
                    image_url = data_item["url"]
                    download_timeout = timeout_val
                    img_headers = auth_headers_for_same_origin(str(image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                    img_res = session.get(image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                    img_res.raise_for_status()
                    image = Image.open(io.BytesIO(img_res.content))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
                except Exception:
                    pass

    # 3. OpenAI Chat Completion format (choices) -> Extract from text content
    if "choices" in res_json and isinstance(res_json["choices"], list) and len(res_json["choices"]) > 0:
        content_text = res_json["choices"][0].get("message", {}).get("content", "")
        if content_text:
            # 3a. Extract URLs (Markdown or Raw)
            urls = re.findall(r"!\[.*?\]\((.*?)\)", content_text)
            if not urls:
                urls = re.findall(r"(https?://[^\s)]+)", content_text)
            
            valid_image_url = None
            for u in urls:
                if str(u).lower().startswith("data:"):
                    continue
                valid_image_url = u
                break
            
            if valid_image_url:
                try:
                    download_timeout = timeout_val
                    img_headers = auth_headers_for_same_origin(str(valid_image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                    img_res = session.get(valid_image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                    img_res.raise_for_status()
                    image = Image.open(io.BytesIO(img_res.content))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
                except Exception:
                    pass

            # 3b. Extract Base64 from text
            try:
                b64_pattern = r"data:image/[^;]+;base64,([a-zA-Z0-9+/=]+)"
                match = re.search(b64_pattern, content_text)
                b64_clean = ""
                if match:
                    b64_clean = match.group(1)
                else:
                    # Clean up markdown/data URI prefix manually
                    temp_clean = re.sub(r"^!\[.*?\]\(", "", content_text.strip())
                    temp_clean = re.sub(r"\)$", "", temp_clean)
                    temp_clean = re.sub(r"^data:image/.+;base64,", "", temp_clean)
                    b64_clean = re.sub(r"\s+", "", temp_clean)
                
                if len(b64_clean) > 100:
                    image_data = base64.b64decode(b64_clean)
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return image
            except Exception:
                pass

    return None
