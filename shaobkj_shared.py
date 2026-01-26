import os
import json
import re
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time
import requests
import io
import base64
from urllib.parse import urlparse
from datetime import datetime
import subprocess
import traceback
import random
import folder_paths

import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except Exception as e:
        print(f"[ComfyUI-shaobkj] Error loading config.json: {e}")

# --- Async Task Manager ---
ASYNC_TASK_FILE = os.path.join(os.path.dirname(__file__), "shaobkj_async_history.json")
async_task_lock = threading.Lock()

def _read_async_tasks():
    if not os.path.exists(ASYNC_TASK_FILE): return {}
    try:
        with open(ASYNC_TASK_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        return tasks if isinstance(tasks, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}

def _write_async_tasks(tasks):
    try:
        # Simple cleanup: keep last 50 tasks
        if len(tasks) > 50:
             # Sort by timestamp if available, else by key
             sorted_keys = sorted(tasks.keys(), key=lambda k: tasks[k].get("submitted_at", "0"))
             keys_to_remove = sorted_keys[:-50]
             for k in keys_to_remove:
                 if tasks[k].get("status") in ["downloaded", "failed"]:
                     del tasks[k]

        with open(ASYNC_TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"[Shaobkj-Async] Write failed: {e}")

def update_async_task(task_id, data):
    with async_task_lock:
        tasks = _read_async_tasks()
        if task_id not in tasks:
            tasks[task_id] = {}
        tasks[task_id].update(data)
        _write_async_tasks(tasks)

def get_async_task(task_id):
    with async_task_lock:
        tasks = _read_async_tasks()
        return tasks.get(task_id)

def get_all_async_tasks():
    with async_task_lock:
        return _read_async_tasks()


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


def resize_and_encode_image(img, long_side):
    """
    Resize PIL image and encode to base64.
    Returns (base64_string, aspect_ratio)
    """
    if img is None:
        return None, 1.0
    
    # Ensure RGB
    if hasattr(img, "mode") and img.mode != "RGB":
        img = img.convert("RGB")
    
    # Calculate aspect ratio
    w, h = img.size
    ratio = w / h
    
    # Use existing resize logic
    img_resized = resize_pil_long_side(img, long_side)
    
    buffered = io.BytesIO()
    img_resized.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str, ratio


def sanitize_text(s, max_len=600):
    t = "" if s is None else str(s)
    t = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[省略]", t)
    t = re.sub(r"[A-Za-z0-9+/=]{200,}", "[省略]", t)
    if len(t) > max_len:
        t = t[:max_len] + "...(省略)"
    return t


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
            # Use stream=True to allow enforcing total timeout during content download
            resp = session.post(
                url,
                headers=headers,
                timeout=current_attempt_timeout,
                verify=verify,
                proxies=proxies,
                stream=True,
                **request_kwargs,
            )
            
            # Enforce total timeout during download if limit is set
            if total_timeout_limit is not None:
                content_chunks = []
                for chunk in resp.iter_content(chunk_size=4096):
                    if time.time() - start_time > total_timeout_limit:
                        resp.close()
                        raise requests.exceptions.ReadTimeout(f"Total execution time exceeded limit of {total_timeout_limit}s during download")
                    if chunk:
                        content_chunks.append(chunk)
                resp._content = b"".join(content_chunks)
                resp._content_consumed = True
            else:
                # If no total limit, read all content to ensure consistent behavior
                # This mimics requests' default behavior when stream=False
                _ = resp.content

        except requests.exceptions.RequestException as e:
            last_exc = e
            # CRITICAL: Never retry on ReadTimeout to prevent double billing
            if isinstance(e, requests.exceptions.ReadTimeout):
                raise
            
            if attempt >= max_retries:
                raise
            # Check if sleep would exceed total timeout
            sleep_time = 2 * attempt
            if total_timeout_limit and (time.time() - start_time + sleep_time) > total_timeout_limit:
                 # If sleeping would timeout, check if we have a response
                 if resp is not None:
                     return resp
                 # No response yet, so we must raise the last exception
                 raise last_exc or requests.exceptions.RequestException("Request failed with no response and retry limit reached")
            time.sleep(sleep_time)
            continue

        if resp.status_code in (500, 502, 503, 504) and attempt < max_retries:
            sleep_time = 2 * attempt
            if total_timeout_limit and (time.time() - start_time + sleep_time) > total_timeout_limit:
                 if resp is not None:
                     return resp
                 # Should not happen here since resp is checked, but for safety
                 raise requests.exceptions.RequestException("Server error and retry timeout reached")
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
                
                # Check for text content containing markdown image with base64
                text_content = part.get("text")
                if text_content:
                    try:
                        # Regex to find markdown image with data URI: ![...](data:image/...;base64,...)
                        match = re.search(r"!\[.*?\]\((data:image/[^;]+;base64,[a-zA-Z0-9+/=]+)\)", text_content)
                        if match:
                            data_uri = match.group(1)
                            if "base64," in data_uri:
                                b64_data = data_uri.split("base64,")[1]
                                image_data = base64.b64decode(b64_data)
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

# --- New Helper Functions for Phase 1 & 2 ---

def smart_pad_images_to_tensor(pil_images):
    """
    Safely convert a list of PIL images to a batch tensor, 
    automatically padding smaller images to the largest size in the batch.
    Prevents errors when mixing aspect ratios/sizes in a batch.
    """
    if not pil_images:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    tensors = []
    max_h = 0
    max_w = 0

    for img in pil_images:
        if img is None: continue
        try:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Convert to float32 numpy array [H, W, 3]
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # To Tensor [1, H, W, 3]
            tensor = torch.from_numpy(img_array)[None,]
            
            if tensor.shape[1] > max_h: max_h = tensor.shape[1]
            if tensor.shape[2] > max_w: max_w = tensor.shape[2]
            tensors.append(tensor)
        except Exception as e:
            print(f"[Shaobkj-Shared] Warning: Skipping broken image: {e}")
            continue

    if not tensors:
        return torch.empty((0, 1, 1, 3), dtype=torch.float32)

    padded_tensors = []
    for tensor in tensors:
        b, h, w, c = tensor.shape
        if h == max_h and w == max_w:
            padded_tensors.append(tensor)
            continue
        
        # Pad: (left, right, top, bottom) for last 2 dims if NCHW, but here we have NHWC
        # Torch F.pad works on last dimensions. 
        # For NHWC, we need to permute to NCHW to pad H and W easily, or be careful.
        # Easier: permute -> pad -> permute back
        tensor_chw = tensor.permute(0, 3, 1, 2) # [1, 3, H, W]
        pad_w = max_w - w
        pad_h = max_h - h
        # Pad right and bottom (filling with black/0)
        padding = (0, pad_w, 0, pad_h) 
        padded_tensor_chw = F.pad(tensor_chw, padding, "constant", 0)
        padded_tensor_hwc = padded_tensor_chw.permute(0, 2, 3, 1) # [1, H, W, 3]
        padded_tensors.append(padded_tensor_hwc)

    return torch.cat(padded_tensors, dim=0)


def robust_download_video(video_url, output_path, max_retries=3, timeout=300, headers=None):
    """
    Attempts to download a video using yt-dlp first, then falls back to system curl.
    Prevents deadlocks by using subprocess with timeout and DEVNULL.
    Supports custom headers for authentication.
    """
    import shutil
    
    print(f"[Shaobkj-Downloader] Downloading: {video_url}")

    # 1. Try yt-dlp (Python library)
    try:
        import yt_dlp
        ydl_opts = {
            'outtmpl': output_path,
            'retries': max_retries,
            'quiet': True,
            'noplaylist': True,
            'merge_output_format': 'mp4',
            'socket_timeout': 30,
            'nocheckcertificate': True,
        }
        if headers:
            ydl_opts['http_headers'] = headers

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            return True
    except ImportError:
        print("[Shaobkj-Downloader] yt-dlp not installed, skipping.")
    except Exception as e:
        print(f"[Shaobkj-Downloader] yt-dlp failed: {e}")
        # Clean up potential partial file
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass

    # 2. Fallback to Curl
    curl_path = shutil.which("curl")
    if curl_path:
        for attempt in range(max_retries):
            try:
                print(f"[Shaobkj-Downloader] Trying curl (attempt {attempt+1})...")
                cmd = [
                    curl_path, "-k", "-L",
                    "--connect-timeout", "20",
                    "--max-time", str(timeout),
                    "-o", output_path,
                    video_url
                ]
                # Add headers
                if headers:
                    for k, v in headers.items():
                        cmd.extend(["-H", f"{k}: {v}"])

                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout + 10,
                    check=True
                )
                if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    return True
            except subprocess.TimeoutExpired:
                print("[Shaobkj-Downloader] Curl timed out.")
            except Exception as e:
                print(f"[Shaobkj-Downloader] Curl failed: {e}")
            time.sleep(2)
    
    return False


def save_local_record(record_type, id_val, remark, source_info):
    """
    Appends a record to a local text file for history tracking.
    """
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        filename = f"{record_type}_history.txt"
        file_path = os.path.join(current_dir, filename)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        remark_str = remark.strip() if remark else "None"
        
        new_record = (
            f"[{timestamp}]\n"
            f"ID:     {id_val}\n"
            f"Remark: {remark_str}\n"
            f"Source: {source_info}\n"
            f"----------------------------------------\n"
        )
        
        old_content = ""
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    old_content = f.read()
            except:
                pass
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_record + old_content)
        return True
    except Exception as e:
        print(f"[Shaobkj-Recorder] Failed to save record: {e}")
        return False
