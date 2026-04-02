import os
import json
import time
import threading
import traceback
import base64
import io
import random
import re
import torch
import numpy as np
from collections import deque
from urllib.parse import urlparse
import folder_paths
from PIL import Image, ImageOps, ImageCms
from server import PromptServer
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.utils import ProgressBar
import nodes
import node_helpers
import comfy

from .shaobkj_shared import (
    get_config_value,
    create_requests_session,
    disable_insecure_request_warnings,
    build_submit_timeout,
    post_json_with_retry,
    auth_headers_for_same_origin,
    resize_and_encode_image,
    resize_pil_long_side,
    extract_image_from_json,
    save_local_record,
    sanitize_text,
    update_async_task,
    get_all_async_tasks,
    pil_to_tensor,
    tensor_to_pil,
    crop_image_to_ratio,
    detect_subject_bbox,
)

def get_closest_aspect_ratio(width, height):
    ratios = {
        "1:1": 1.0,
        "16:9": 16/9,
        "9:16": 9/16,
        "4:3": 4/3,
        "3:4": 3/4,
        "3:2": 3/2,
        "2:3": 2/3,
        "21:9": 21/9,
        "9:21": 9/21
    }
    target = width / height
    closest_ratio = "1:1"
    min_diff = float('inf')
    
    for r_str, r_val in ratios.items():
        diff = abs(target - r_val)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = r_str
            
    return closest_ratio

def get_first_list_value(value, default=None):
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value if value is not None else default

def flatten_generic_values(value):
    flat_list = []
    if isinstance(value, list):
        for item in value:
            flat_list.extend(flatten_generic_values(item))
    else:
        flat_list.append(value)
    return flat_list

def tensor_value_to_pil_list(value):
    pil_images = []
    if isinstance(value, list):
        for item in value:
            pil_images.extend(tensor_value_to_pil_list(item))
    elif isinstance(value, torch.Tensor):
        if value.dim() == 4:
            for i in range(value.shape[0]):
                pil_images.append(Image.fromarray(np.clip(255. * value[i].cpu().numpy(), 0, 255).astype(np.uint8)))
        elif value.dim() == 3:
            pil_images.append(Image.fromarray(np.clip(255. * value.cpu().numpy(), 0, 255).astype(np.uint8)))
    return pil_images

def normalize_image_rounds(value):
    if isinstance(value, list):
        return [tensor_value_to_pil_list(item) for item in value]
    return [tensor_value_to_pil_list(value)]

GROUPED_CONCURRENT_BUFFER = {}
GROUPED_CONCURRENT_BUFFER_LOCK = threading.Lock()
GROUPED_CONCURRENT_IDLE_SECONDS = 0.8

def run_grouped_concurrent_tasks(tasks, max_workers, interval):
    total_tasks = len(tasks)
    if total_tasks <= 0:
        return

    print(f"[ComfyUI-shaobkj] [Grouped-Concurrent] 检测到上游已完成，开始发送 {total_tasks} 组任务。", flush=True)

    success_count = 0
    fail_count = 0
    completed_count = 0
    failure_reasons = []
    lock = threading.Lock()

    def on_task_done(fut, tid):
        nonlocal success_count, fail_count, completed_count

        try:
            fut.result()
            is_success = True
            error_msg = None
        except Exception as e:
            is_success = False
            error_msg = str(e)

        with lock:
            completed_count += 1
            if is_success:
                success_count += 1
                status_str = "完成"
            else:
                fail_count += 1
                failure_reasons.append(f"{tid}: {error_msg}")
                status_str = f"失败，原因: {error_msg}"

            print(f"[ComfyUI-shaobkj-BG] [Grouped-Concurrent] 进度: {completed_count}/{total_tasks} | 成功: {success_count} | 失败: {fail_count} | 任务 {tid} {status_str}", flush=True)

            if completed_count == total_tasks:
                summary = f"[ComfyUI-shaobkj-BG] [Grouped-Concurrent] 任务已全部完成。总计: {total_tasks} | 成功: {success_count} | 失败: {fail_count}"
                if fail_count > 0:
                    summary += f" | 失败详情: {'; '.join(failure_reasons)}"
                print(summary, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, task_data in enumerate(tasks):
            if interval > 0 and i > 0:
                time.sleep(interval)
            future = executor.submit(run_concurrent_task_internal, task_data)
            future.add_done_callback(lambda f, t=task_data["task_id"]: on_task_done(f, t))
            futures.append(future)
        for future in futures:
            try:
                future.result()
            except Exception:
                pass

def schedule_grouped_concurrent_flush(unique_id, total_output_count, max_workers, interval):
    with GROUPED_CONCURRENT_BUFFER_LOCK:
        state = GROUPED_CONCURRENT_BUFFER.setdefault(unique_id, {"tasks": [], "version": 0})
        version = state["version"]

    def delayed_flush():
        time.sleep(GROUPED_CONCURRENT_IDLE_SECONDS)
        with GROUPED_CONCURRENT_BUFFER_LOCK:
            current_state = GROUPED_CONCURRENT_BUFFER.get(unique_id)
            if not current_state or current_state["version"] != version:
                return
            tasks_to_send = list(current_state["tasks"])
            current_state["tasks"].clear()

        if not tasks_to_send:
            return

        if total_output_count > 0:
            tasks_to_send = tasks_to_send[:total_output_count]

        run_grouped_concurrent_tasks(tasks_to_send, max_workers, interval)

    t = threading.Thread(target=delayed_flush)
    t.daemon = True
    t.start()

# ----------------------------------------------------------------------------
# Background Worker (Async Sender Logic)
# ----------------------------------------------------------------------------

def run_concurrent_task_internal(data):
    # Use provided task_id or generate new one
    task_id_local = data.get("task_id")
    if not task_id_local:
        task_id_local = f"task_{int(time.time())}_{random.randint(1000,9999)}"
    
    print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Starting concurrent task {task_id_local}...")
    
    # Initial status update
    update_async_task(task_id_local, {
        "status": "running",
        "submitted_at": int(time.time()),
        "prompt": data.get("prompt", "")[:50] + "...",
        "type": "edit"
    })

    try:
        # Parse common params
        api_key = data.get("api_key")
        api_url_base = data.get("api_url", "https://yhmx.work")
        model = data.get("model", "gemini-3-pro-image-preview")
        use_proxy = data.get("use_proxy", False)
        resolution = data.get("resolution", "1k")
        prompt = data.get("prompt", "")
        aspect_ratio = data.get("aspect_ratio", "原图1比例")
        long_side = int(data.get("long_side", 1280))
        wait_time = int(data.get("wait_time", 0))
        seed_val = int(data.get("seed", 0))
        save_path_input = data.get("save_path", "")
        save_format_input = data.get("save_format", "JPEG (默认95%)")
        accept_mode = data.get("accept_mode", "智能模式")
        subject_text = data.get("subject_text", "")

        if not api_key:
             raise ValueError("API Key is required")

        # Collect Images
        pil_images = []
        
        # Process Uploaded Image
        image_name = data.get("image_name")
        if image_name:
             try:
                 p = folder_paths.get_annotated_filepath(image_name)
                 if p and os.path.exists(p):
                     img = Image.open(p)
                     img = ImageOps.exif_transpose(img)
                     pil_images.append(img)
             except Exception as e:
                 print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Error loading uploaded image: {e}")

        # Process Additional Uploads (from JS dynamic inputs)
        additional_images = data.get("additional_images", [])
        for item in additional_images:
            if isinstance(item, dict) and item.get("value"):
                 try:
                     p = folder_paths.get_annotated_filepath(item.get("value"))
                     if p and os.path.exists(p):
                         img = Image.open(p)
                         img = ImageOps.exif_transpose(img)
                         pil_images.append(img)
                 except Exception:
                     pass

        # Process Tensor Images (already PIL)
        if "tensor_images" in data:
            pil_images.extend(data["tensor_images"])

        if not pil_images:
             raise ValueError("No valid images found (Check uploads or connections).")

        first_image_ratio = None
        first_image_size = None
        if pil_images:
            w0, h0 = pil_images[0].size
            if w0 > 0 and h0 > 0:
                first_image_ratio = float(w0) / float(h0)
                first_image_size = (int(w0), int(h0))
        subject_crop_ratio = None
        if isinstance(subject_text, str) and subject_text.strip():
            if aspect_ratio == "原图1比例" and first_image_size is not None:
                subject_crop_ratio = get_closest_aspect_ratio(first_image_size[0], first_image_size[1])
            elif aspect_ratio and aspect_ratio != "Free":
                subject_crop_ratio = str(aspect_ratio)

        def get_target_size_local(res, ar, first_size):
            target_map = {"1k": 1024, "2k": 2048, "4k": 4096}
            target = target_map.get(str(res).lower(), 1024)
            ratio_str = str(ar)
            if ratio_str == "原图1比例" and first_size is not None:
                ratio_str = get_closest_aspect_ratio(first_size[0], first_size[1])
            if ratio_str == "原图1比例" or ratio_str == "Free":
                return target, target
            if ":" in ratio_str:
                try:
                    a, b = ratio_str.split(":", 1)
                    aw = float(a)
                    ah = float(b)
                    if aw > 0 and ah > 0:
                        r = aw / ah
                        if r >= 1.0:
                            w = target
                            h = max(1, int(round(target / r)))
                        else:
                            h = target
                            w = max(1, int(round(target * r)))
                        return int(w), int(h)
                except Exception:
                    pass
            return target, target

        def adjust_image_aspect(pil_image):
            if pil_image is None:
                return None
            if aspect_ratio != "原图1比例" or first_image_ratio is None:
                return pil_image
            target_w, target_h = get_target_size_local(resolution, aspect_ratio, first_image_size)
            if target_w <= 0 or target_h <= 0:
                return pil_image
            w, h = pil_image.size
            if w <= 0 or h <= 0:
                return pil_image
            target_ratio = float(target_w) / float(target_h)
            src_ratio = float(w) / float(h)
            if abs(src_ratio - target_ratio) > 0.001:
                if src_ratio > target_ratio:
                    new_w = max(1, int(round(h * target_ratio)))
                    left = max(0, int((w - new_w) // 2))
                    pil_image = pil_image.crop((left, 0, left + new_w, h))
                else:
                    new_h = max(1, int(round(w / target_ratio)))
                    top = max(0, int((h - new_h) // 2))
                    pil_image = pil_image.crop((0, top, w, top + new_h))
            if pil_image.size != (int(target_w), int(target_h)):
                pil_image = pil_image.resize((int(target_w), int(target_h)), Image.LANCZOS)
            return pil_image

        def encode_input_image(pil_image):
            if pil_image is None:
                return None, None
            img = resize_pil_long_side(pil_image, long_side)
            if subject_crop_ratio:
                detect_timeout = 30 if wait_time <= 0 else max(5, min(30, int(wait_time)))
                bbox = detect_subject_bbox(img, subject_text.strip(), api_url_base, api_key, use_proxy, detect_timeout)
                img = crop_image_to_ratio(img, subject_crop_ratio, bbox)
            if "PNG" in str(save_format_input):
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8"), "image/png"
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8"), "image/jpeg"

        if model == "智能加载":
            resolution_key = str(resolution).lower()
            if resolution_key == "2k":
                model = "gemini-3-pro-image-preview-2k"
            elif resolution_key == "4k":
                model = "gemini-3-pro-image-preview-4k"
            else:
                model = "gemini-3-pro-image-preview"

        # Prepare Request
        base_origin = str(api_url_base).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        # Fix: Remove Authorization header for Gemini to improve proxy compatibility (align with ModeHub)
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
        # Force generation instruction
        final_prompt = str(prompt) + "\n\n(Generate an image based on this description)"
        parts = [{"text": final_prompt}]
        
        image_b64_list = [] # Store for fallback
        
        for img in pil_images:
            try:
                b64_str, mime_type = encode_input_image(img)
                if b64_str:
                    image_b64_list.append(b64_str)
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_str
                        }
                    })
            except Exception as e:
                print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Error encoding image: {e}")

        # Seed Logic
        safe_seed = seed_val
        if safe_seed < 0:
            safe_seed = random.randint(0, 2147483647)
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.7, 
                "seed": safe_seed, 
                "responseModalities": ["TEXT", "IMAGE"]
            }
        }
        payload["generationConfig"]["imageConfig"] = {"imageSize": str(resolution).upper()}

        target_aspect_ratio = aspect_ratio
        if target_aspect_ratio == "原图1比例" and len(pil_images) > 0:
             # Calculate from first image (assuming image_1 is first in list)
             w, h = pil_images[0].size
             target_aspect_ratio = get_closest_aspect_ratio(w, h)
             print(f"[ComfyUI-shaobkj] Calculated aspect ratio from image 1 ({w}x{h}): {target_aspect_ratio}")

        if target_aspect_ratio != "原图1比例" and target_aspect_ratio != "Free":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(target_aspect_ratio)

        # Debug: Print Image Config to verify Aspect Ratio
        print(f"[ComfyUI-shaobkj] [Debug] Sending Image Config: {payload.get('generationConfig', {}).get('imageConfig', {})}")

        # Send Request
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(use_proxy))
        submit_timeout = build_submit_timeout(wait_time)
        
        def decode_b64_image(b64_str):
            try:
                image_data = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(image_data))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            except Exception:
                return None

        def download_url_image(image_url):
            try:
                img_headers = auth_headers_for_same_origin(str(image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                img_res = session.get(image_url, verify=False, timeout=60, proxies=proxies, headers=img_headers)
                img_res.raise_for_status()
                image = Image.open(io.BytesIO(img_res.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            except Exception:
                return None

        def extract_image_by_mode(res_json):
            if accept_mode == "智能模式":
                return extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60)
            if not isinstance(res_json, dict):
                return None
            if accept_mode == "URL":
                data_list = res_json.get("data")
                if isinstance(data_list, list):
                    for item in data_list:
                        if isinstance(item, dict) and isinstance(item.get("url"), str):
                            img = download_url_image(item.get("url"))
                            if img:
                                return img
                choices = res_json.get("choices")
                if isinstance(choices, list) and choices:
                    content_text = choices[0].get("message", {}).get("content", "")
                    if isinstance(content_text, str) and content_text:
                        urls = re.findall(r"!\[.*?\]\((.*?)\)", content_text)
                        if not urls:
                            urls = re.findall(r"(https?://[^\s)]+)", content_text)
                        for u in urls:
                            if isinstance(u, str) and not u.lower().startswith("data:"):
                                img = download_url_image(u)
                                if img:
                                    return img
                candidates = res_json.get("candidates")
                if isinstance(candidates, list) and candidates:
                    for cand in candidates:
                        content = cand.get("content") if isinstance(cand, dict) else None
                        parts = content.get("parts") if isinstance(content, dict) else None
                        if not isinstance(parts, list):
                            continue
                        for part in parts:
                            if not isinstance(part, dict):
                                continue
                            text_content = part.get("text")
                            if isinstance(text_content, str) and text_content:
                                urls = re.findall(r"!\[.*?\]\((.*?)\)", text_content)
                                if not urls:
                                    urls = re.findall(r"(https?://[^\s)]+)", text_content)
                                for u in urls:
                                    if isinstance(u, str) and not u.lower().startswith("data:"):
                                        img = download_url_image(u)
                                        if img:
                                            return img
                return None
            if accept_mode == "B64":
                candidates = res_json.get("candidates")
                if isinstance(candidates, list) and candidates:
                    for cand in candidates:
                        content = cand.get("content") if isinstance(cand, dict) else None
                        parts = content.get("parts") if isinstance(content, dict) else None
                        if not isinstance(parts, list):
                            continue
                        for part in parts:
                            if not isinstance(part, dict):
                                continue
                            inline = part.get("inlineData") or part.get("inline_data")
                            if isinstance(inline, dict) and inline.get("data"):
                                img = decode_b64_image(inline.get("data"))
                                if img:
                                    return img
                            text_content = part.get("text")
                            if isinstance(text_content, str) and text_content:
                                m = re.search(r"data:image/[^;]+;base64,([a-zA-Z0-9+/=]+)", text_content)
                                if m:
                                    img = decode_b64_image(m.group(1))
                                    if img:
                                        return img
                data_list = res_json.get("data")
                if isinstance(data_list, list):
                    for item in data_list:
                        if isinstance(item, dict) and item.get("b64_json"):
                            img = decode_b64_image(item.get("b64_json"))
                            if img:
                                return img
                choices = res_json.get("choices")
                if isinstance(choices, list) and choices:
                    content_text = choices[0].get("message", {}).get("content", "")
                    if isinstance(content_text, str) and content_text:
                        m = re.search(r"data:image/[^;]+;base64,([a-zA-Z0-9+/=]+)", content_text)
                        if m:
                            img = decode_b64_image(m.group(1))
                            if img:
                                return img
                return None
            return None

        # A. Helper Functions for Fallback
        def try_openai_fallback():
            openai_base = base_origin[:-3] if base_origin.endswith("/v1") else base_origin
            openai_url = f"{openai_base}/v1/chat/completions"
            openai_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            openai_content = [{"type": "text", "text": final_prompt}]
            for b64_str in image_b64_list:
                openai_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})
            openai_payload = {
                "model": model,
                "messages": [{"role": "user", "content": openai_content}],
                "temperature": 0.7,
                "seed": safe_seed,
            }
            try:
                openai_resp = post_json_with_retry(
                    session,
                    openai_url,
                    headers=openai_headers,
                    payload=openai_payload,
                    timeout=submit_timeout,
                    proxies=proxies,
                    verify=False,
                )
                openai_resp.raise_for_status()
                openai_json = openai_resp.json()
                return extract_image_by_mode(openai_json)
            except Exception as e:
                print(f"[ComfyUI-shaobkj] [Concurrent-Sender] OpenAI Fallback failed: {e}")
                return None

        def get_task_id_from_headers(resp):
            headers_map = getattr(resp, "headers", {}) or {}
            tid = headers_map.get("X-Task-Id") or headers_map.get("Task-Id") or headers_map.get("task_id") or headers_map.get("task-id")
            if tid: return tid
            location = headers_map.get("Location") or headers_map.get("location")
            if isinstance(location, str) and location:
                m = re.search(r"/([^/]+)/?$", location.strip())
                if m: return m.group(1)
            return None

        print(f"[ComfyUI-shaobkj] {task_id_local}: Sending request...")
        
        # B. Request with 524 Handling and Retry
        response = post_json_with_retry(
            session,
            url,
            headers=headers,
            payload=payload,
            timeout=submit_timeout,
            proxies=proxies,
            verify=False
        )


        
        if response.status_code not in (200, 201, 202):
            print(f"[ComfyUI-shaobkj] [Concurrent-Sender] {task_id_local}: API Error Status: {response.status_code}")
            # Try to read error body for logging
            try:
                print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Error Body: {response.text[:200]}")
            except Exception:
                pass
        
        response.raise_for_status()
        
        try:
            res_json = response.json()
        except (json.JSONDecodeError, ValueError):
             # Handle Empty Response Body
             raw_text = response.text
             if not raw_text or not raw_text.strip():
                 print(f"[ComfyUI-shaobkj] [Concurrent-Sender] {task_id_local}: Warning: Empty response body (HTTP {response.status_code})")
                 task_id = get_task_id_from_headers(response)
                 if task_id:
                     res_json = {"id": task_id}
                 else:
                     print(f"[ComfyUI-shaobkj] [Concurrent-Sender] {task_id_local}: Trying OpenAI fallback due to empty response")
                     extracted_img = try_openai_fallback()
                     if extracted_img:
                         res_json = {"status": "success", "fallback": True}
                     else:
                         raise RuntimeError(f"API Error: Empty response body (HTTP {response.status_code})")
             else:
                 raise

        # Verify response content
        extracted_img = None
        if "fallback" in res_json:
             pass # Already extracted
        else:
             if not res_json or (not res_json.get("candidates") and not res_json.get("id") and not res_json.get("name") and not res_json.get("data")):
                print(f"[ComfyUI-shaobkj] {task_id_local}: Warning - Empty or invalid response from API.")

             # Extract Result using shared helper
             extracted_img = extract_image_by_mode(res_json)
        
        remote_task_id = None
        
        if not extracted_img:
             remote_task_id = res_json.get("id") or res_json.get("task_id")
             if not remote_task_id and "name" in res_json:
                 # Operation name as ID
                 remote_task_id = res_json["name"]
             if not remote_task_id and "data" in res_json:
                 remote_task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
             if not remote_task_id:
                 # Check headers one last time
                 remote_task_id = get_task_id_from_headers(response)

             if remote_task_id:
                 print(f"[ComfyUI-shaobkj] {task_id_local}: Polling remote task {remote_task_id}...")
                 poll_url = f"{url}/{remote_task_id}"
                 poll_timeout_val = 86400 if wait_time == 0 else wait_time
                 start_poll = time.time()
                 fail_count = 0
                 
                 while True:
                     if (time.time() - start_poll) > poll_timeout_val:
                         raise RuntimeError("Timeout polling")
                     
                     time.sleep(2)
                     try:
                        poll_resp = session.get(poll_url, headers=headers, params={"_t": int(time.time()*1000)}, verify=False, proxies=proxies, timeout=30)
                        fail_count = 0
                        if poll_resp.status_code == 200:
                            poll_json = poll_resp.json()
                            extracted_img = extract_image_by_mode(poll_json)
                            if extracted_img:
                                break
                            
                            status = poll_json.get("status") or poll_json.get("task_status")
                            if status in ["FAILED", "ERROR"]:
                                raise RuntimeError(f"Remote Task failed: {status}")
                     except Exception as e:
                         fail_count += 1
                         print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Polling error: {e}")
                         if fail_count > 10:
                             raise

        # Save Result
        if extracted_img:
            final_img = adjust_image_aspect(extracted_img)
            # Determine Format
            save_params = {"format": "JPEG", "quality": 95}
            ext = ".jpg"
            
            if save_format_input and "PNG" in save_format_input:
                save_params = {"format": "PNG"}
                ext = ".png"
            elif save_format_input and "WEBP" in save_format_input:
                save_params = {"format": "WEBP", "lossless": True}
                ext = ".webp"

            # Determine Filename
            custom_filename = data.get("output_filename")
            if custom_filename:
                base_name = os.path.splitext(os.path.basename(str(custom_filename)))[0]
                filename = f"{base_name}{ext}"
            else:
                filename = f"concurrent_edit_{int(time.time())}_{random.randint(1000,9999)}{ext}"
            
            # Determine output directory
            out_dir = folder_paths.get_output_directory()
            if save_path_input and isinstance(save_path_input, str) and save_path_input.strip():
                custom_dir = save_path_input.strip()
                # Check if absolute or relative
                if not os.path.isabs(custom_dir):
                    custom_dir = os.path.join(out_dir, custom_dir)
                
                try:
                    os.makedirs(custom_dir, exist_ok=True)
                    out_dir = custom_dir
                except Exception as e:
                    print(f"[ComfyUI-shaobkj] [Concurrent-Sender] Failed to create custom dir {custom_dir}, using default. Error: {e}")

            out_path = os.path.join(out_dir, filename)
            
            # Check for overwrite and append suffix if needed
            counter = 1
            original_out_path = out_path
            original_base_name = os.path.splitext(filename)[0]
            
            while os.path.exists(out_path):
                new_filename = f"{original_base_name}_{counter}{ext}"
                out_path = os.path.join(out_dir, new_filename)
                filename = new_filename # Update filename variable for logging/return
                counter += 1
            
            final_img.save(out_path, **save_params)
            
            print(f"[ComfyUI-shaobkj] [Concurrent-Sender] {task_id_local}: Success! Saved to {out_path}")
            
            # Record success in async manager
            update_async_task(task_id_local, {
                "status": "success",
                "image_path": out_path,
                "completed_at": int(time.time())
            })
            
            # Also record in local log file
            save_local_record("Concurrent_Edit", str(remote_task_id or task_id_local), "Success", api_url_base)
            
            PromptServer.instance.send_sync("shaobkj.concurrent.success", {"task_id": task_id_local, "filename": filename, "path": out_path})
            
            return task_id_local # Return success ID
        else:
            brief = sanitize_text(json.dumps(res_json, ensure_ascii=False))
            raise RuntimeError(f"No image data found in response: {brief}")

    except Exception as e:
        err_msg = str(e)
        # Simplify common API errors
        if "401" in err_msg or "Unauthorized" in err_msg or "invalid_api_key" in err_msg:
             err_msg = "❌ 错误：API Key 无效或未授权 (401 Unauthorized)。请检查您的 API Key 是否正确。"
        elif "404" in err_msg or "Not Found" in err_msg:
             err_msg = "❌ 错误：API 地址或模型未找到 (404 Not Found)。请检查 API 地址和模型名称。"
        elif "429" in err_msg or "Too Many Requests" in err_msg or "quota" in err_msg.lower():
             err_msg = "❌ 错误：API 配额耗尽或请求过于频繁 (429 Too Many Requests)。"
        elif "500" in err_msg or "Internal Server Error" in err_msg:
             err_msg = "❌ 错误：API 服务端内部错误 (500 Internal Server Error)。"
        elif "504" in err_msg or "Gateway Time-out" in err_msg:
             err_msg = "❌ 错误：请求超时 (504 Gateway Time-out)。服务器处理时间过长。"
        elif "Total execution time exceeded limit" in err_msg:
             err_msg = f"❌ 错误：等待超时。任务执行时间超过了设定的限制。"
        elif "Read timed out" in err_msg or "Connect timed out" in err_msg:
             err_msg = f"❌ 错误：网络连接超时。网络响应慢。"
             
        print(f"[ComfyUI-shaobkj] [Concurrent-Sender] {task_id_local}: {err_msg}")
        
        # Record failure in async manager
        update_async_task(task_id_local, {
            "status": "failed",
            "error": err_msg,
            "completed_at": int(time.time())
        })

        traceback.print_exc()
        # Raise to let the executor know it failed
        raise RuntimeError(err_msg)


# ----------------------------------------------------------------------------
# API Route (Backward Compatibility)
# ----------------------------------------------------------------------------

@PromptServer.instance.routes.post("/shaobkj/concurrent/submit")
async def api_concurrent_submit(request):
    try:
        json_data = await request.json()
        
        # Resolve image path
        image_name = json_data.get("image_name")
        if image_name:
            image_path = folder_paths.get_annotated_filepath(image_name)
            json_data["image_path"] = image_path
        
        # Start background thread
        t = threading.Thread(target=run_concurrent_task_internal, args=(json_data,))
        t.daemon = True
        t.start()
        
        return web.json_response({"status": "success", "message": "Task started in background"})
        
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


# ----------------------------------------------------------------------------
# Node A: Sender (Async)
# ----------------------------------------------------------------------------

class Shaobkj_ConcurrentImageEdit_Sender:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "编辑描述，支持多行；推荐：每行一条提示词"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image-preview",
                        "智能加载",
                    ],
                    {"default": "gemini-3-pro-image-preview", "tooltip": "模型选择或智能加载；推荐：gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k", "tooltip": "输出分辨率档位；推荐：1k"}),
                "图片比例": (["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"], {"default": "原图1比例", "tooltip": "输出画面比例；推荐：原图1比例"}),
                "接收模式": (["智能模式", "URL", "B64"], {"default": "智能模式", "tooltip": "API 返回内容处理方式；推荐：智能模式"}),
                "主体文本": ("STRING", {"default": "", "multiline": False, "tooltip": "主体识别裁切关键词；推荐：留空"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "Batch拆分模式": ("BOOLEAN", {"default": True, "tooltip": "是否拆分批次提交；推荐：开启"}),
                "Batch对齐方式": (["循环补全(Max)", "裁切对齐(Min)"], {"default": "循环补全(Max)", "tooltip": "批次对齐策略；推荐：循环补全(Max)"}),
                "保存路径": ("STRING", {"default": "Shaobkj_Concurrent", "multiline": False, "tooltip": "相对输出目录的子路径；推荐：Shaobkj_Concurrent"}),
                "保存格式": (["JPEG (默认95%)", "PNG (无损)", "WEBP (无损)"], {"default": "JPEG (默认95%)", "tooltip": "输出保存格式；推荐：JPEG (默认95%)"}),
                "最大并发数": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "后台最大同时执行任务数；推荐：5"}),
                "并发间隔": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "批量任务提交间隔(秒)；推荐：1.0"}),
            },
            "optional": {
                 "文件名来源": ("STRING", {"forceInput": True, "multiline": False, "dynamicPrompts": False, "tooltip": "用于输出命名的文件名来源；推荐：留空"}),
                 "image_1": ("IMAGE", {"tooltip": "输入图像1；推荐：连接参考图"}),
                 "image_2": ("IMAGE", {"tooltip": "输入图像2；推荐：可选"}),
                 "image_3": ("IMAGE", {"tooltip": "输入图像3；推荐：可选"}),
                 "image_4": ("IMAGE", {"tooltip": "输入图像4；推荐：可选"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("API响应", "状态")
    FUNCTION = "submit_task"
    CATEGORY = "🤖shaobkj-APIbox"
    OUTPUT_NODE = True

    def submit_task(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 接收模式, 主体文本, 保存路径, seed, **kwargs):
        # Unwrap parameters because INPUT_IS_LIST = True wraps everything in lists
        # We assume common parameters are same for all items (take first), OR we should support batching them too.
        # For simplicity, let's take the first item for "global" settings, but support batching for Prompts and Images.
        
        def get_val(v, default=None):
            if isinstance(v, list) and len(v) > 0:
                return v[0]
            return v
        
        api_key_val = get_val(API密钥)
        api_url_val = get_val(API地址)
        model_val = get_val(模型选择)
        use_proxy_val = get_val(使用系统代理)
        resolution_val = get_val(分辨率)
        # prompt is special, might be list of strings
        prompts_val = 提示词 # Keep as list if it is list
        aspect_ratio_val = get_val(图片比例)
        accept_mode_val = get_val(接收模式)
        subject_text_val = get_val(主体文本)
        save_path_val = get_val(保存路径)
        
        # kwargs handling (also wrapped in lists)
        # We need to reconstruct kwargs to be clean
        clean_kwargs = {}
        for k, v in kwargs.items():
            if k == "unique_id": continue
            clean_kwargs[k] = v

        long_side_val = int(get_val(kwargs.get("输入图像-长边设置", [1280])))
        wait_time_val = 0 # Default to infinite wait
        seed_val = int(get_val(seed))
        batch_split_val = get_val(kwargs.get("Batch拆分模式", [True]))
        batch_align_val = get_val(kwargs.get("Batch对齐方式", ["循环补全(Max)"]))
        submit_interval_val = float(get_val(kwargs.get("并发间隔", [1.0])))
        max_workers_val = int(get_val(kwargs.get("最大并发数", [5])))
        save_format_val = get_val(kwargs.get("保存格式", ["JPEG (默认95%)"]))
        filename_source_val = kwargs.get("文件名来源", None) # Keep as list
        
        # 0. Pre-check
        if not api_key_val or str(api_key_val).strip() == "":
            raise ValueError("❌ 错误：API Key 不能为空")
        if not api_url_val or str(api_url_val).strip() == "":
            raise ValueError("❌ 错误：API 地址不能为空")
        
        # 1. Prepare Base Data
        base_task_id = f"task_{int(time.time())}_{random.randint(1000,9999)}"
        
        base_data = {
            "api_key": api_key_val,
            "api_url": api_url_val,
            "model": model_val,
            "use_proxy": use_proxy_val,
            "resolution": resolution_val,
            "prompt": prompts_val, # Might be list
            "aspect_ratio": aspect_ratio_val,
            "accept_mode": accept_mode_val,
            "subject_text": subject_text_val,
            "long_side": long_side_val,
            "wait_time": wait_time_val,
            "seed": seed_val,
            "save_path": save_path_val,
            "save_format": save_format_val
        }

        # Helper to normalize inputs to a flat list of tensors
        def normalize_image_input(val):
            flat_list = []
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, list):
                         flat_list.extend(normalize_image_input(item))
                    elif isinstance(item, torch.Tensor):
                        # item is [B,H,W,C]
                        for i in range(item.shape[0]):
                            flat_list.append(item[i]) # Store as [H,W,C] tensors
            elif isinstance(val, torch.Tensor):
                 for i in range(val.shape[0]):
                    flat_list.append(val[i])
            return flat_list

        # Helper to normalize generic list inputs (recursively flatten)
        def normalize_list_input(val):
            flat_list = []
            if isinstance(val, list):
                for item in val:
                    flat_list.extend(normalize_list_input(item))
            else:
                flat_list.append(val)
            return flat_list

        # 2. Logic Branch: Batch Split vs Single Request
        
        task_list = []
        
        if not batch_split_val:
            # --- Legacy Mode: All images in one request ---
            data = base_data.copy()
            # If prompt is list, join them or take first? Legacy usually takes one string.
            if isinstance(data["prompt"], list):
                data["prompt"] = data["prompt"][0] if data["prompt"] else ""
            
            data["task_id"] = base_task_id
            data["tensor_images"] = []
            
            # Collect Images (Flatten all batches)
            for k, v in clean_kwargs.items():
                if k.startswith("image_"):
                    tensors = normalize_image_input(v)
                    for t in tensors:
                        if t.dim() == 3:
                            pil_img = Image.fromarray(np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8))
                            data["tensor_images"].append(pil_img)
            
            # Handle Filename
            f_list = []
            if filename_source_val:
                if isinstance(filename_source_val, list):
                     f_list = [str(x) for x in filename_source_val]
                else:
                     f_list = [str(filename_source_val)]
            if f_list:
                data["output_filename"] = f_list[0]

            task_list.append(data)

        else:
            # --- Batch Split Mode: One request per aligned item ---
            # 1. Identify Inputs and Normalize them
            normalized_inputs = {}
            for k, v in clean_kwargs.items():
                if k.startswith("image_"):
                    normalized_inputs[k] = normalize_image_input(v)
            
            sorted_keys = sorted(normalized_inputs.keys()) # image_1, image_2...
            
            # 2. Determine Max Batch Size
            batch_sizes = [len(v) for v in normalized_inputs.values()]
            
            # Debugging Info
            debug_msg = "[Shaobkj-Debug] Batch Inputs: "
            for k, v in normalized_inputs.items():
                debug_msg += f"{k}={len(v)}, "
            
            # Handle prompt list
            prompts = []
            raw_prompts = normalize_list_input(prompts_val)
            for p in raw_prompts:
                 if p is not None:
                     prompts.append(str(p))
            
            if not prompts:
                prompts = [""] # Fallback

            if len(prompts) > 1:
                batch_sizes.append(len(prompts))
                debug_msg += f"Prompts={len(prompts)}"
            else:
                debug_msg += "Prompt=Single"
            
            # Handle filename source
            filename_list = []
            if filename_source_val:
                raw_list = normalize_list_input(filename_source_val)
                for item in raw_list:
                    if isinstance(item, str) and "\n" in item:
                        filename_list.extend([x.strip() for x in item.split("\n") if x.strip()])
                    elif item is not None and str(item).strip() != "":
                        filename_list.append(str(item))
                
                if filename_list:
                    batch_sizes.append(len(filename_list))
                    debug_msg += f"Filenames={len(filename_list)}"
                else:
                    debug_msg += "Filenames=None"
            
            if not batch_sizes:
                final_batch_size = 1
            elif batch_align_val == "裁切对齐(Min)":
                final_batch_size = min(batch_sizes)
            else:
                final_batch_size = max(batch_sizes)

            print(f"{debug_msg} => Mode={batch_align_val} => Final Batch Size: {final_batch_size}")
            
            for i in range(final_batch_size):
                sub_task_id = f"{base_task_id}_{i}"
                sub_data = base_data.copy()
                sub_data["task_id"] = sub_task_id
                sub_data["tensor_images"] = []
                
                # Assign Prompt
                if prompts:
                    sub_data["prompt"] = prompts[i % len(prompts)]
                
                # Assign Filename
                if filename_list:
                    sub_data["output_filename"] = filename_list[i % len(filename_list)]
                
                # Assign Images (Aligned Slicing)
                for k in sorted_keys:
                    tensor_list = normalized_inputs[k]
                    if tensor_list:
                        idx = i % len(tensor_list)
                        t = tensor_list[idx]
                        if t.dim() == 3:
                            pil_img = Image.fromarray(np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8))
                            sub_data["tensor_images"].append(pil_img)
                
                task_list.append(sub_data)

        # 3. Execute Tasks in Background (Fire-and-Forget)
        
        total_tasks = len(task_list)
        print(f"[ComfyUI-shaobkj] Preparing to run {total_tasks} tasks in background...")
        
        def background_batch_monitor(tasks, max_workers, split_mode, interval):
            total_tasks = len(tasks)
            print(f"[ComfyUI-shaobkj-BG] [Concurrent-Sender] Background monitor started for {total_tasks} tasks. (Max Workers: {max_workers}, Interval: {interval}s)")
            
            success_count = 0
            fail_count = 0
            completed_count = 0
            failure_reasons = []
            lock = threading.Lock()
            
            def on_task_done(fut, tid):
                nonlocal success_count, fail_count, completed_count
                
                try:
                    fut.result()
                    is_success = True
                    error_msg = None
                except Exception as e:
                    is_success = False
                    error_msg = str(e)

                with lock:
                    completed_count += 1
                    if is_success:
                        success_count += 1
                        status_str = "完成"
                    else:
                        fail_count += 1
                        failure_reasons.append(f"{tid}: {error_msg}")
                        status_str = f"失败，原因: {error_msg}"
                    
                    print(f"[ComfyUI-shaobkj-BG] [Concurrent-Sender] 进度: {completed_count}/{total_tasks} | 成功: {success_count} | 失败: {fail_count} | 任务 {tid} {status_str}")

                    if completed_count == total_tasks:
                        summary = f"[ComfyUI-shaobkj-BG] [Concurrent-Sender] 任务已全部完成。总计: {total_tasks} | 成功: {success_count} | 失败: {fail_count}"
                        if fail_count > 0:
                            summary += f" | 失败详情: {'; '.join(failure_reasons)}"
                        print(summary)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, task_data in enumerate(tasks):
                    # Apply delay if needed
                    if split_mode and interval > 0 and i > 0:
                        time.sleep(interval)
                    
                    future = executor.submit(run_concurrent_task_internal, task_data)
                    # Use lambda to capture task_id immediately
                    future.add_done_callback(lambda f, t=task_data["task_id"]: on_task_done(f, t))

        # Launch Thread
        # We can use max_workers=5 or similar
        max_workers = max_workers_val
        if max_workers <= 0:
            max_workers = 1000 # "Unlimited" effectively
        
        t = threading.Thread(target=background_batch_monitor, args=(task_list, max_workers, batch_split_val, submit_interval_val))
        t.daemon = True
        t.start()
        
        msg = f"已成功提交 {total_tasks} 个任务到后台运行。\n请查看控制台日志或输出目录监控进度。\n前端可继续操作。"
        print(f"[ComfyUI-shaobkj] {msg}")
        
        # Return immediate feedback
        # Note: generated_ids is now just the list of IDs we *submitted*
        generated_ids = [t["task_id"] for t in task_list]
        status_list = ["Submitted (Background)" for _ in task_list]
        
        return (generated_ids, status_list)

class Shaobkj_GroupedConcurrentImageEdit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "编辑描述，支持多行；推荐：每行一条提示词"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                        "gemini-3.1-flash-image-preview",
                        "智能加载",
                    ],
                    {"default": "gemini-3-pro-image-preview", "tooltip": "模型选择或智能加载；推荐：gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k", "tooltip": "输出分辨率档位；推荐：1k"}),
                "图片比例": (["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"], {"default": "原图1比例", "tooltip": "输出画面比例；推荐：原图1比例"}),
                "接收模式": (["智能模式", "URL", "B64"], {"default": "智能模式", "tooltip": "API 返回内容处理方式；推荐：智能模式"}),
                "出图数量": ("INT", {"default": 5, "min": 1, "max": 999999, "step": 1, "tooltip": "最终发送的组任务总数/生成图像总数；推荐：5"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "Batch对齐方式": (["循环补全(Max)", "裁切对齐(Min)"], {"default": "循环补全(Max)", "tooltip": "批次对齐策略；推荐：循环补全(Max)"}),
                "保存路径": ("STRING", {"default": "Shaobkj_Concurrent", "multiline": False, "tooltip": "相对输出目录的子路径；推荐：Shaobkj_Concurrent"}),
                "保存格式": (["JPEG (默认95%)", "PNG (无损)", "WEBP (无损)"], {"default": "JPEG (默认95%)", "tooltip": "输出保存格式；推荐：JPEG (默认95%)"}),
                "最大并发数": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "后台最大同时执行任务数；推荐：5"}),
                "并发间隔": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "批量任务提交间隔(秒)；推荐：1.0"}),
            },
            "optional": {
                "文件名来源": ("STRING", {"forceInput": True, "multiline": False, "dynamicPrompts": False, "tooltip": "用于输出命名的文件名来源；推荐：留空"}),
                "image_1": ("IMAGE", {"tooltip": "输入图像1；推荐：连接参考图"}),
                "image_2": ("IMAGE", {"tooltip": "输入图像2；推荐：可选"}),
                "image_3": ("IMAGE", {"tooltip": "输入图像3；推荐：可选"}),
                "image_4": ("IMAGE", {"tooltip": "输入图像4；推荐：可选"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("API响应", "状态")
    FUNCTION = "submit_grouped_task"
    CATEGORY = "🤖shaobkj-APIbox"
    OUTPUT_NODE = True

    def submit_grouped_task(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 接收模式, 出图数量, 保存路径, seed, **kwargs):
        api_key_val = get_first_list_value(API密钥)
        api_url_val = get_first_list_value(API地址)
        model_val = get_first_list_value(模型选择)
        use_proxy_val = get_first_list_value(使用系统代理)
        resolution_val = get_first_list_value(分辨率)
        prompts_val = 提示词
        aspect_ratio_val = get_first_list_value(图片比例)
        accept_mode_val = get_first_list_value(接收模式)
        output_count_val = max(1, int(get_first_list_value(出图数量, 5)))
        save_path_val = get_first_list_value(保存路径)
        long_side_val = int(get_first_list_value(kwargs.get("输入图像-长边设置", [1280])))
        seed_val = int(get_first_list_value(seed, 0))
        batch_align_val = get_first_list_value(kwargs.get("Batch对齐方式", ["循环补全(Max)"]))
        submit_interval_val = float(get_first_list_value(kwargs.get("并发间隔", [1.0]), 1.0))
        max_workers_val = int(get_first_list_value(kwargs.get("最大并发数", [5]), 5))
        save_format_val = get_first_list_value(kwargs.get("保存格式", ["JPEG (默认95%)"]))
        filename_source_val = kwargs.get("文件名来源", None)
        unique_id_val = str(get_first_list_value(kwargs.get("unique_id", "")) or "grouped_concurrent_default")

        if not api_key_val or str(api_key_val).strip() == "":
            raise ValueError("❌ 错误：API Key 不能为空")
        if not api_url_val or str(api_url_val).strip() == "":
            raise ValueError("❌ 错误：API 地址不能为空")

        base_task_id = f"grouped_{int(time.time())}_{random.randint(1000,9999)}"
        base_data = {
            "api_key": api_key_val,
            "api_url": api_url_val,
            "model": model_val,
            "use_proxy": use_proxy_val,
            "resolution": resolution_val,
            "aspect_ratio": aspect_ratio_val,
            "accept_mode": accept_mode_val,
            "subject_text": "",
            "long_side": long_side_val,
            "wait_time": 0,
            "seed": seed_val,
            "save_path": save_path_val,
            "save_format": save_format_val,
        }

        normalized_inputs = {}
        for key, value in kwargs.items():
            if key in {"unique_id", "文件名来源", "输入图像-长边设置", "Batch对齐方式", "并发间隔", "最大并发数", "保存格式"}:
                continue
            if key.startswith("image_"):
                normalized_inputs[key] = normalize_image_rounds(value)

        sorted_keys = sorted(normalized_inputs.keys())

        prompts = []
        for item in flatten_generic_values(prompts_val):
            if item is not None:
                prompts.append(str(item))
        if not prompts:
            prompts = [""]

        filename_list = []
        if filename_source_val:
            for item in flatten_generic_values(filename_source_val):
                if isinstance(item, str) and "\n" in item:
                    filename_list.extend([x.strip() for x in item.split("\n") if x.strip()])
                elif item is not None and str(item).strip() != "":
                    filename_list.append(str(item))

        batch_sizes = [len(prompts)]
        for image_rounds in normalized_inputs.values():
            batch_sizes.append(len(image_rounds))
        if filename_list:
            batch_sizes.append(len(filename_list))

        if not batch_sizes:
            final_batch_size = 1
        elif batch_align_val == "裁切对齐(Min)":
            final_batch_size = min(batch_sizes)
        else:
            final_batch_size = max(batch_sizes)

        if final_batch_size <= 0:
            print("[ComfyUI-shaobkj] [Grouped-Concurrent] 未收到可提交的数据，跳过本次执行。")
            return ([], [])

        task_list = []
        for i in range(final_batch_size):
            sub_task_id = f"{base_task_id}_{i}"
            sub_data = base_data.copy()
            sub_data["task_id"] = sub_task_id
            sub_data["prompt"] = prompts[i % len(prompts)] if prompts else ""
            sub_data["tensor_images"] = []

            if filename_list:
                sub_data["output_filename"] = filename_list[i % len(filename_list)]

            for key in sorted_keys:
                image_rounds = normalized_inputs[key]
                if not image_rounds:
                    continue
                round_images = image_rounds[i % len(image_rounds)]
                if round_images:
                    sub_data["tensor_images"].extend(round_images)

            task_list.append(sub_data)

        with GROUPED_CONCURRENT_BUFFER_LOCK:
            state = GROUPED_CONCURRENT_BUFFER.setdefault(unique_id_val, {"tasks": [], "version": 0})
            state["tasks"].extend(task_list)
            state["version"] += 1

        max_workers = max_workers_val if max_workers_val > 0 else 1000
        schedule_grouped_concurrent_flush(unique_id_val, output_count_val, max_workers, submit_interval_val)

        generated_ids = [task["task_id"] for task in task_list]
        status_list = ["Buffered" for _ in generated_ids]

        return (generated_ids, status_list)


# ----------------------------------------------------------------------------
# Node C: Load Batch Images From Path
# ----------------------------------------------------------------------------
class Shaobkj_Load_Image_Path:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "从输入目录选择图像文件；推荐：选择一张图像"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    FUNCTION = "load_image"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"❌ 错误：文件不存在: {image}")
        if not os.path.isfile(image_path):
            raise ValueError(f"❌ 错误：不是有效文件: {image}")

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")

        filename = os.path.basename(image_path)
        return (image, mask, filename)


class Shaobkj_Load_Batch_Images:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False, "placeholder": "输入文件夹路径 (如 C:\\images)", "tooltip": "图片所在文件夹路径；推荐：填写有效路径"}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "限制加载数量，0为不限制；推荐：0"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "从第几个文件开始加载；推荐：0"}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "每次运行都重新加载；推荐：关闭"}),
                "sort_method": (["numerical", "alphabetical", "date"], {"default": "numerical", "tooltip": "文件排序方式；推荐：numerical"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "文件名")
    FUNCTION = "load_images"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def load_images(self, directory, image_load_cap=0, start_index=0, load_always=False, sort_method="numerical"):
        folder_path = directory
        if not folder_path or not os.path.exists(folder_path):
             raise ValueError(f"❌ 错误：文件夹路径不存在: {folder_path}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        file_list = []
        
        try:
            for f in os.listdir(folder_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(folder_path, f)
                    if os.path.isfile(full_path):
                        file_list.append(full_path)
        except Exception as e:
            raise ValueError(f"❌ 错误：读取文件夹失败: {e}")

        if sort_method == "numerical":
             def natural_sort_key(s):
                 parts = re.split('([0-9]+)', s)
                 processed = []
                 for text in parts:
                     if text.isdigit():
                         processed.append((0, int(text)))
                     else:
                         cleaned = text.strip(" ()[]{}-_")
                         processed.append((1, cleaned.lower()))
                 return processed
             
             try:
                 file_list.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
             except Exception as e:
                 print(f"[Shaobkj-Loader] Numerical sort failed: {e}. Fallback to default sort.")
                 file_list.sort()
        elif sort_method == "alphabetical":
             file_list.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_method == "date":
             file_list.sort(key=lambda x: os.path.getmtime(x))
        else:
             file_list.sort()
        
        if start_index > 0:
            if start_index >= len(file_list):
                 file_list = []
            else:
                 file_list = file_list[start_index:]
        
        if image_load_cap > 0:
            file_list = file_list[:image_load_cap]

        if not file_list:
             raise ValueError(f"❌ 错误：文件夹为空或筛选后无有效图片: {folder_path}")

        print(f"[Shaobkj-Loader] Found {len(file_list)} images in {folder_path}")
        
        images_out = []
        masks_out = []
        filenames_out = []
        
        for file_path in file_list:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                image = img.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                if 'A' in img.getbands():
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
                images_out.append(image)
                masks_out.append(mask)
                filenames_out.append(os.path.basename(file_path))
                
            except Exception as e:
                print(f"[Shaobkj-Loader] Error loading {file_path}: {e}")

        if not images_out:
             raise ValueError("No images loaded successfully.")

        return (images_out, masks_out, filenames_out)

class Shaobkj_Image_Save:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入图像；推荐：连接上游图像输出"}),
                "保存路径": ("STRING", {"default": "Shaobkj_Save", "multiline": False, "tooltip": "相对输出目录的子路径；推荐：Shaobkj_Save"}),
                "保存格式": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"], {"default": "png", "tooltip": "保存格式；推荐：png"}),
                "模式": (["保持原色", "RGB 颜色", "CMYK 颜色", "Lab 颜色"], {"default": "保持原色", "tooltip": "颜色模式；推荐：保持原色"}),
                "文件名": ("STRING", {"default": "image", "multiline": False, "tooltip": "保存文件名(不含扩展名)；推荐：image"}),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1, "tooltip": "输出图像DPI；推荐：300"}),
                "质量": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1, "tooltip": "JPG 质量(1-100)；推荐：100"}),
                "自定义尺寸": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否", "tooltip": "是否启用中心裁切到指定宽高"}),
                "宽": ("INT", {"default": 1024, "min": 1, "max": 200000, "step": 1, "tooltip": "自定义输出宽度"}),
                "高": ("INT", {"default": 1024, "min": 1, "max": 200000, "step": 1, "tooltip": "自定义输出高度"}),
                "预览": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "是否在界面显示预览；推荐：开启"}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_image"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    OUTPUT_NODE = True

    def save_image(self, 图像, 保存路径, 保存格式, 模式, 文件名, dpi, 质量, 自定义尺寸, 宽, 高, 预览):
        def resolve_cmyk_profile_path():
            candidates = []
            windir = os.environ.get("WINDIR", r"C:\Windows")
            color_dir = os.path.join(windir, "System32", "spool", "drivers", "color")
            if os.path.isdir(color_dir):
                try:
                    files = [f for f in os.listdir(color_dir) if f.lower().endswith((".icc", ".icm"))]
                except Exception:
                    files = []
                priority_keywords = ["cmyk", "coated", "swop", "fogra", "japan", "pso"]
                files_sorted = sorted(
                    files,
                    key=lambda n: (
                        0 if any(k in n.lower() for k in priority_keywords) else 1,
                        n.lower(),
                    ),
                )
                for name in files_sorted:
                    candidates.append(os.path.join(color_dir, name))
            for path in candidates:
                try:
                    profile = ImageCms.getOpenProfile(path)
                    color_space = str(getattr(getattr(profile, "profile", None), "xcolor_space", "")).upper()
                    if "CMYK" in color_space:
                        return path
                except Exception:
                    continue
            return None

        def convert_rgb_to_cmyk(pil_img_rgb):
            profile_path = resolve_cmyk_profile_path()
            if profile_path:
                try:
                    src_profile = ImageCms.createProfile("sRGB")
                    dst_profile = ImageCms.getOpenProfile(profile_path)
                    converted = ImageCms.profileToProfile(
                        pil_img_rgb.convert("RGB"),
                        src_profile,
                        dst_profile,
                        outputMode="CMYK",
                        renderingIntent=0,
                    )
                    return converted, False
                except Exception:
                    pass
            return pil_img_rgb.convert("CMYK"), True

        images = 图像
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)

        output_root = folder_paths.get_output_directory()
        out_dir = output_root
        if isinstance(保存路径, str) and 保存路径.strip():
            custom_dir = 保存路径.strip()
            if not os.path.isabs(custom_dir):
                custom_dir = os.path.join(output_root, custom_dir)
            os.makedirs(custom_dir, exist_ok=True)
            out_dir = custom_dir

        fmt_label = str(保存格式).strip().lower() if 保存格式 is not None else "png"
        valid_formats = {"png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"}
        extension = fmt_label if fmt_label in valid_formats else "png"
        ext = f".{extension}"

        base_name = str(文件名).strip() if 文件名 is not None else ""
        base_name = os.path.splitext(os.path.basename(base_name))[0]
        if not base_name:
            base_name = "image"

        filenames = []
        out_paths = []
        warning_sent = False
        crop_warning_sent = False
        cmyk_profile_warning_sent = False
        preview_images = []
        for i in range(images.shape[0]):
            img_tensor = images[i]
            t = img_tensor
            if isinstance(t, torch.Tensor) and t.dim() == 4:
                t = t[0]
            if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
                t = t.permute(1, 2, 0)
            arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
            arr = np.clip(arr, 0.0, 1.0)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if bool(自定义尺寸):
                target_w = max(1, int(宽))
                target_h = max(1, int(高))
                h_src, w_src = int(arr.shape[0]), int(arr.shape[1])
                crop_w = min(target_w, w_src)
                crop_h = min(target_h, h_src)
                if (crop_w != target_w or crop_h != target_h) and not crop_warning_sent:
                    PromptServer.instance.send_sync(
                        "shaobkj.image_save.warning",
                        {
                            "message": f"⚠️ 自定义尺寸({target_w}x{target_h})超过原图尺寸，已按中心裁切到可用尺寸({crop_w}x{crop_h})。"
                        },
                    )
                    crop_warning_sent = True
                x0 = max(0, (w_src - crop_w) // 2)
                y0 = max(0, (h_src - crop_h) // 2)
                arr = arr[y0:y0 + crop_h, x0:x0 + crop_w, :]
            if extension == "png":
                if arr.shape[-1] >= 4:
                    img_arr = (arr[:, :, :4] * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(img_arr, mode="RGBA")
                else:
                    alpha = np.ones((arr.shape[0], arr.shape[1], 1), dtype=arr.dtype)
                    rgba = np.concatenate([arr[:, :, :3], alpha], axis=2)
                    img_arr = (rgba * 255.0).astype(np.uint8)
                    pil_img = Image.fromarray(img_arr, mode="RGBA")
            else:
                if arr.shape[-1] >= 4:
                    rgb = arr[:, :, :3]
                    alpha = np.clip(arr[:, :, 3:4], 0.0, 1.0)
                    white = np.ones_like(rgb)
                    comp = np.clip(rgb + (1.0 - alpha) * white, 0.0, 1.0)
                    img_arr = (comp * 255.0).astype(np.uint8)
                else:
                    img_arr = (arr[:, :, :3] * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(img_arr, mode="RGB")

            selected_mode = str(模式).strip() if 模式 is not None else "保持原色"
            if selected_mode == "RGB 颜色":
                pil_img = pil_img.convert("RGB")
            elif selected_mode == "CMYK 颜色":
                converted_img, fallback_used = convert_rgb_to_cmyk(pil_img.convert("RGB"))
                pil_img = converted_img
                if fallback_used and not cmyk_profile_warning_sent:
                    PromptServer.instance.send_sync(
                        "shaobkj.image_save.warning",
                        {"message": "⚠️ 未找到可用 CMYK ICC 配置文件，已使用基础 CMYK 转换，可能存在显示偏差。"},
                    )
                    cmyk_profile_warning_sent = True
            elif selected_mode == "Lab 颜色":
                pil_img = pil_img.convert("LAB")

            if extension in ("png", "webp", "gif", "bmp") and pil_img.mode in ("CMYK", "LAB"):
                if not warning_sent:
                    PromptServer.instance.send_sync(
                        "shaobkj.image_save.warning",
                        {
                            "message": f"⚠️ 当前保存格式 {extension.upper()} 不支持 {selected_mode}，已自动转为 RGB 保存。"
                        },
                    )
                    warning_sent = True
                pil_img = pil_img.convert("RGB")

            filename = f"{base_name}{ext}"
            out_path = os.path.join(out_dir, filename)
            counter = 1
            while os.path.exists(out_path):
                filename = f"{base_name}_{counter}{ext}"
                out_path = os.path.join(out_dir, filename)
                counter += 1
            if extension in ("jpg", "jpeg"):
                pil_img.save(out_path, format="JPEG", quality=int(质量), dpi=(int(dpi), int(dpi)))
            elif extension == "png":
                pil_img.save(out_path, format="PNG", dpi=(int(dpi), int(dpi)))
            elif extension == "webp":
                pil_img.save(out_path, format="WEBP", quality=int(质量))
            elif extension == "gif":
                pil_img.save(out_path, format="GIF")
            elif extension == "tiff":
                if pil_img.mode == "CMYK":
                    pil_img.save(out_path, format="TIFF", compression="raw", dpi=(int(dpi), int(dpi)))
                else:
                    pil_img.save(out_path, format="TIFF", compression="tiff_deflate", dpi=(int(dpi), int(dpi)))
            elif extension == "bmp":
                pil_img.save(out_path, format="BMP")
            else:
                pil_img.save(out_path, format="PNG", dpi=(int(dpi), int(dpi)))
            filenames.append(filename)
            out_paths.append(out_path)
            preview_base = os.path.splitext(filename)[0]
            preview_img = None
            try:
                with Image.open(out_path) as saved_img:
                    preview_img = ImageOps.exif_transpose(saved_img).convert("RGB")
            except Exception:
                preview_img = pil_img.convert("RGB")
            preview_images.append((preview_img, preview_base))

        if 预览:
            preview_entries = []
            preview_dir = os.path.join(output_root, "Shaobkj_Preview")
            os.makedirs(preview_dir, exist_ok=True)
            preview_subfolder = "Shaobkj_Preview"
            for idx, (preview_img, preview_base) in enumerate(preview_images):
                preview_name = f"{preview_base}.png"
                preview_path = os.path.join(preview_dir, preview_name)
                counter = 1
                while os.path.exists(preview_path):
                    preview_name = f"{preview_base}_{counter}.png"
                    preview_path = os.path.join(preview_dir, preview_name)
                    counter += 1
                preview_img.save(preview_path, format="PNG")
                preview_entries.append({"filename": preview_name, "subfolder": preview_subfolder, "type": "output"})

            return {"ui": {"images": preview_entries}}
        return {}

class Shaobkj_FourWayRepair_HD:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入图像；推荐：待修复图像"}),
                "模型": ("MODEL", {"tooltip": "使用的模型；推荐：与工程一致模型"}),
                "VAE": ("VAE", {"tooltip": "VAE 模型；推荐：与模型匹配"}),
                "正面条件": ("CONDITIONING", {"tooltip": "正面提示词条件；推荐：连接正面条件"}),
                "负面条件": ("CONDITIONING", {"tooltip": "负面提示词条件；推荐：连接负面条件"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "随机种子；推荐：0"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "采样步数；推荐：20"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "CFG 引导强度；推荐：8.0"}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "采样器类型；推荐：默认值"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "调度器类型；推荐：默认值"}),
                "修补带宽百分比": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01, "tooltip": "修补带宽比例；推荐：0.15"}),
                "软边比例": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "修补软边比例；推荐：0.5"}),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "去噪强度；推荐：0.75"}),
                "启用分块": ("BOOLEAN", {"default": True, "tooltip": "是否启用分块处理；推荐：开启"}),
                "分块尺寸": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64, "tooltip": "分块大小；推荐：1024"}),
                "分块重叠": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8, "tooltip": "分块重叠像素；推荐：64"}),
                "噪声遮罩": ("BOOLEAN", {"default": True, "tooltip": "是否使用噪声遮罩；推荐：开启"}),
            },
            "optional": {
                "遮罩": ("MASK", {"tooltip": "可选遮罩；推荐：需要区域时连接"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("图像", "latent")
    FUNCTION = "repair"
    CATEGORY = "🤖shaobkj-APIbox"

    def repair(self, 图像, 模型, VAE, 正面条件, 负面条件, seed, steps, cfg, 采样器, 调度器, 修补带宽百分比, 软边比例, denoise, 启用分块, 分块尺寸, 分块重叠, 噪声遮罩=True, 遮罩=None):
        def build_soft_band_mask(h, w, band_w, soft_ratio, axis, device):
            band_w = max(1, int(band_w))
            soft_ratio = float(soft_ratio)
            feather = max(1, int(round(band_w * soft_ratio)))
            if axis == "v":
                start = max(0, w // 2 - band_w // 2)
                end = min(w, start + band_w)
                x = torch.arange(w, device=device)
                dist = torch.where(x < start, start - x, torch.where(x >= end, x - (end - 1), torch.zeros_like(x)))
                mask_x = torch.clamp(1.0 - dist.float() / float(feather), 0.0, 1.0)
                return mask_x.view(1, w).repeat(h, 1)
            start = max(0, h // 2 - band_w // 2)
            end = min(h, start + band_w)
            y = torch.arange(h, device=device)
            dist = torch.where(y < start, start - y, torch.where(y >= end, y - (end - 1), torch.zeros_like(y)))
            mask_y = torch.clamp(1.0 - dist.float() / float(feather), 0.0, 1.0)
            return mask_y.view(h, 1).repeat(1, w)

        def inpaint_once(pixels_hw, mask_hw):
            pixels_bhwc = pixels_hw.unsqueeze(0)
            h_local, w_local = int(pixels_bhwc.shape[1]), int(pixels_bhwc.shape[2])
            mask_bchw = mask_hw.reshape((1, 1, h_local, w_local))
            x = (pixels_bhwc.shape[1] // 8) * 8
            y = (pixels_bhwc.shape[2] // 8) * 8
            mask_interp = torch.nn.functional.interpolate(mask_bchw, size=(pixels_bhwc.shape[1], pixels_bhwc.shape[2]), mode="bilinear").clamp(0.0, 1.0)
            orig_pixels = pixels_bhwc
            pixels = orig_pixels.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask_interp = mask_interp[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
            m = (1.0 - mask_interp).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] -= 0.5
                pixels[:, :, :, i] *= m
                pixels[:, :, :, i] += 0.5
            concat_latent = VAE.encode(pixels)
            orig_latent = VAE.encode(orig_pixels)
            latent = {"samples": orig_latent}
            if 噪声遮罩:
                latent["noise_mask"] = mask_interp
            pos = node_helpers.conditioning_set_values(正面条件, {"concat_latent_image": concat_latent, "concat_mask": mask_interp})
            neg = node_helpers.conditioning_set_values(负面条件, {"concat_latent_image": concat_latent, "concat_mask": mask_interp})
            sampled = nodes.common_ksampler(模型, int(seed), int(steps), float(cfg), 采样器, 调度器, pos, neg, latent, denoise=float(denoise))[0]
            decoded = VAE.decode(sampled["samples"])
            return decoded[0]

        def get_mask_input(h, w, device):
            if 遮罩 is None:
                return None
            m_in = 遮罩
            if isinstance(m_in, torch.Tensor) and m_in.dim() == 4:
                m_in = m_in[0, 0] if m_in.shape[1] == 1 else m_in[0]
            if isinstance(m_in, torch.Tensor) and (m_in.shape[0] != h or m_in.shape[1] != w):
                m_in = torch.nn.functional.interpolate(m_in.reshape((1, 1, m_in.shape[-2], m_in.shape[-1])), size=(h, w), mode="bilinear").squeeze(0).squeeze(0)
            return m_in.to(device=device, dtype=torch.float32)

        def build_tile_indices(length, tile, overlap):
            tile = max(1, int(tile))
            overlap = max(0, int(overlap))
            if tile >= length:
                return [0]
            stride = max(1, tile - overlap)
            indices = []
            pos = 0
            while True:
                if pos + tile >= length:
                    indices.append(max(0, length - tile))
                    break
                indices.append(pos)
                pos += stride
            return indices

        def build_tile_weight(h, w, y0, x0, tile_h, tile_w, overlap):
            overlap = max(0, int(overlap))
            if overlap == 0:
                return torch.ones((tile_h, tile_w), device=img.device, dtype=torch.float32)
            wx = torch.ones((tile_w,), device=img.device, dtype=torch.float32)
            wy = torch.ones((tile_h,), device=img.device, dtype=torch.float32)
            if x0 > 0:
                ramp = torch.linspace(0.0, 1.0, steps=min(overlap, tile_w), device=img.device)
                wx[:ramp.shape[0]] = ramp
            if x0 + tile_w < w:
                ramp = torch.linspace(1.0, 0.0, steps=min(overlap, tile_w), device=img.device)
                wx[-ramp.shape[0]:] = torch.minimum(wx[-ramp.shape[0]:], ramp)
            if y0 > 0:
                ramp = torch.linspace(0.0, 1.0, steps=min(overlap, tile_h), device=img.device)
                wy[:ramp.shape[0]] = ramp
            if y0 + tile_h < h:
                ramp = torch.linspace(1.0, 0.0, steps=min(overlap, tile_h), device=img.device)
                wy[-ramp.shape[0]:] = torch.minimum(wy[-ramp.shape[0]:], ramp)
            return wy.view(tile_h, 1) * wx.view(1, tile_w)

        def inpaint_tiled(pixels_hw, mask_full, tile_size, overlap):
            h, w = int(pixels_hw.shape[0]), int(pixels_hw.shape[1])
            ys = build_tile_indices(h, tile_size, overlap)
            xs = build_tile_indices(w, tile_size, overlap)
            acc = torch.zeros_like(pixels_hw)
            acc_w = torch.zeros((h, w), device=pixels_hw.device, dtype=torch.float32)
            for y0 in ys:
                for x0 in xs:
                    y1 = min(h, y0 + tile_size)
                    x1 = min(w, x0 + tile_size)
                    tile = pixels_hw[y0:y1, x0:x1, :]
                    mask_tile = mask_full[y0:y1, x0:x1]
                    tile_out = inpaint_once(tile, mask_tile)
                    weight = build_tile_weight(h, w, y0, x0, tile.shape[0], tile.shape[1], overlap)
                    acc[y0:y1, x0:x1, :] += tile_out * weight.unsqueeze(-1)
                    acc_w[y0:y1, x0:x1] += weight
            acc = acc / acc_w.clamp(min=1e-6).unsqueeze(-1)
            return acc

        images = 图像
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)
        batch = []
        latents = []
        for img in images:
            if isinstance(img, torch.Tensor) and img.dim() == 4:
                img = img[0]
            img = torch.clamp(img, 0.0, 1.0)
            h, w = int(img.shape[0]), int(img.shape[1])
            bw = max(1, int(round(min(h, w) * float(修补带宽百分比))))
            shifted = torch.roll(img, shifts=(h // 2, w // 2), dims=(0, 1))
            img_stage = shifted
            mask_in = get_mask_input(h, w, img.device)
            mask_v = build_soft_band_mask(h, w, bw, 软边比例, "v", img.device)
            if mask_in is not None:
                mask_v = torch.maximum(mask_v, mask_in)
            mask_h = build_soft_band_mask(h, w, bw, 软边比例, "h", img.device)
            if mask_in is not None:
                mask_h = torch.maximum(mask_h, mask_in)
            if 启用分块 and max(h, w) > int(分块尺寸):
                tile = int(分块尺寸)
                overlap = int(分块重叠)
                img_stage = inpaint_tiled(img_stage, mask_v, tile, overlap)
                img_stage = inpaint_tiled(img_stage, mask_h, tile, overlap)
            else:
                img_stage = inpaint_once(img_stage, mask_v)
                img_stage = inpaint_once(img_stage, mask_h)
            out = torch.roll(img_stage, shifts=(-h // 2, -w // 2), dims=(0, 1))
            out = torch.clamp(out, 0.0, 1.0)
            batch.append(out)
            latent_samples = VAE.encode(out.unsqueeze(0))
            latents.append(latent_samples)
        latent_out = {"samples": torch.cat(latents, dim=0)} if latents else {"samples": torch.empty((0,))}
        return (torch.stack(batch, dim=0), latent_out)

class Shaobkj_Fixed_Seed:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "tooltip": "固定随机种子；推荐：0"})
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    CATEGORY = "🤖shaobkj-APIbox"

    def get_seed(self, seed):
        return (int(seed),)

    @classmethod
    def IS_CHANGED(cls, seed):
        return seed

# ----------------------------------------------------------------------------
# Node Registration
# ----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Shaobkj_ConcurrentImageEdit_Sender": Shaobkj_ConcurrentImageEdit_Sender,
    "Shaobkj_GroupedConcurrentImageEdit": Shaobkj_GroupedConcurrentImageEdit,
    "Shaobkj_Load_Image_Path": Shaobkj_Load_Image_Path,
    "Shaobkj_Load_Batch_Images": Shaobkj_Load_Batch_Images,
    "Shaobkj_Image_Save": Shaobkj_Image_Save,
    "Shaobkj_FourWayRepair_HD": Shaobkj_FourWayRepair_HD,
    "Shaobkj_Fixed_Seed": Shaobkj_Fixed_Seed
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_ConcurrentImageEdit_Sender": "🤖并发-编辑-图像驱动",
    "Shaobkj_GroupedConcurrentImageEdit": "🧩组合并发",
    "Shaobkj_Load_Image_Path": "🤖加载图像",
    "Shaobkj_Load_Batch_Images": "🤖批量加载图像(路径)",
    "Shaobkj_Image_Save": "🤖图像保存",
    "Shaobkj_FourWayRepair_HD": "🤖四方修复高清",
    "Shaobkj_Fixed_Seed": "🤖固定随机种子"
}
