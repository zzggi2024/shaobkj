import os
import json
import torch
import numpy as np
from PIL import Image
import io
import base64
import re
import random
import time
import traceback
import concurrent.futures
import threading
from urllib.parse import urlparse
import pandas as pd
import folder_paths
from server import PromptServer

import torch.nn.functional as F

from .shaobkj_shared import (
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    extract_image_from_json,
    auth_headers_for_same_origin,
    get_config_value,
    pil_to_tensor,
    tensor_to_pil,
    post_json_with_retry,
    save_local_record,
    update_async_task,
    get_all_async_tasks,
    resize_pil_long_side,
    crop_image_to_ratio,
    detect_subject_bbox,
)
from comfy.utils import ProgressBar


def sanitize_text(s, max_len=1200):
    t = "" if s is None else str(s)
    t = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[省略]", t)
    t = re.sub(r"[A-Za-z0-9+/=]{200,}", "[省略]", t)
    if len(t) > max_len:
        t = t[:max_len] + "...(省略)"
    return t


def encode_pil_image(img, use_png, quality=85):
    if img is None:
        return None
    buffered = io.BytesIO()
    if use_png:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img.save(buffered, format="PNG")
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="JPEG", quality=int(quality))
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class Shaobkj_APINode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "生成内容描述，支持多行；推荐：每行一条提示词"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                        "智能加载",
                    ],
                    {"default": "gemini-3-pro-image-preview", "tooltip": "模型选择或智能加载；推荐：gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k", "tooltip": "输出分辨率档位；推荐：1k"}),
                "图片比例": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"],
                    {"default": "原图1比例", "tooltip": "输出画面比例；推荐：原图1比例"},
                ),
                "接收模式": (["智能模式", "URL", "B64"], {"default": "智能模式", "tooltip": "API 返回内容处理方式；推荐：智能模式"}),
                "主体文本": ("STRING", {"default": "", "multiline": False, "tooltip": "主体识别裁切关键词；推荐：留空"}),
                "保存格式": (["JPEG (默认95%)", "PNG (无损)", "WEBP (无损)"], {"default": "JPEG (默认95%)", "tooltip": "输出保存格式；推荐：JPEG (默认95%)"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待；推荐：180"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False, "tooltip": "API 申请入口；推荐：默认地址"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "API响应")
    FUNCTION = "generate_image"
    CATEGORY = "🤖shaobkj-APIbox"

    def snap_to_aspect_ratio(self, ratio):
        """
        Snaps a float ratio (width/height) to the nearest standard aspect ratio string.
        """
        # Standard ratios map: float_value -> string_representation
        standards = {
            1.0: "1:1",
            4/3: "4:3",
            3/4: "3:4",
            3/2: "3:2",
            2/3: "2:3",
            16/9: "16:9",
            9/16: "9:16",
            21/9: "21:9",
            9/21: "9:21"
        }
        
        # Find closest
        closest_dist = float('inf')
        closest_str = "1:1"
        
        for r_val, r_str in standards.items():
            dist = abs(ratio - r_val)
            if dist < closest_dist:
                closest_dist = dist
                closest_str = r_str
                
        return closest_str

    def generate_image(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词, 图片比例, 接收模式, 主体文本, 保存格式, 等待时间, seed, **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        gemini_base = base_origin[:-3] if base_origin.endswith("/v1") else base_origin
        api_origin = urlparse(gemini_base).netloc
        resolution = 分辨率
        prompt = 提示词
        aspect_ratio = 图片比例
        accept_mode = 接收模式
        subject_text = str(主体文本).strip() if isinstance(主体文本, str) else ""
        save_format_input = 保存格式
        long_side_limit = int(kwargs.get("输入图像-长边设置", 1280))
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        seed_value = seed

        temperature = 0.7

        if not api_key:
            raise ValueError("API Key is required.")

        model = 模型选择
        if model == "智能加载":
            resolution_key = str(resolution).lower()
            if resolution_key == "2k":
                model = "gemini-3-pro-image-preview-2k"
            elif resolution_key == "4k":
                model = "gemini-3-pro-image-preview-4k"
            else:
                model = "gemini-3-pro-image-preview"

        # Fix: Remove Authorization header for Gemini to improve proxy compatibility (align with ModeHub)
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        url = f"{gemini_base}/v1beta/models/{model}:generateContent"

        # Force image generation by appending instruction
        final_prompt = str(prompt) + "\n\n(Generate an image based on this description)"
        parts = [{"text": final_prompt}]

        # Process input images if any
        image_inputs = []
        for k, v in kwargs.items():
            if k.startswith("image_") and v is not None:
                image_inputs.append((k, v))
        
        # Sort images by key name
        image_inputs.sort(key=lambda x: int(x[0].split("_")[1]))
        
        first_image_ratio = None
        first_image_pil = None
        
        image_b64_list = []
        for idx, (name, tensor) in enumerate(image_inputs):
            pil_img = tensor_to_pil(tensor)
            use_png = "PNG" in str(save_format_input)
            pil_img = resize_pil_long_side(pil_img, long_side_limit)
            ratio_for_crop = None
            if subject_text:
                if aspect_ratio == "原图1比例":
                    w, h = pil_img.size
                    if h > 0:
                        ratio_for_crop = self.snap_to_aspect_ratio(float(w) / float(h))
                elif aspect_ratio and aspect_ratio != "Free":
                    ratio_for_crop = str(aspect_ratio)
            if subject_text and ratio_for_crop:
                detect_timeout = 30 if timeout_val is None else max(5, min(30, int(timeout_val)))
                bbox = detect_subject_bbox(pil_img, subject_text, base_origin, api_key, 使用系统代理, detect_timeout)
                pil_img = crop_image_to_ratio(pil_img, ratio_for_crop, bbox)
            if pil_img is None:
                continue
            b64_str = encode_pil_image(pil_img, use_png, quality=85)
            ratio = float(pil_img.size[0]) / float(pil_img.size[1])
            if idx == 0:
                first_image_ratio = ratio
                first_image_pil = tensor_to_pil(tensor)
            if b64_str:
                image_b64_list.append(b64_str)
                parts.append({
                    "inline_data": {
                        "mime_type": "image/png" if use_png else "image/jpeg",
                        "data": b64_str
                    }
                })

        payload = {"contents": [{"role": "user", "parts": parts}]}
        safe_seed = int(seed_value)
        if safe_seed < 0:
            safe_seed = random.randint(0, 2147483647)
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647

        payload["generationConfig"] = {"temperature": temperature, "seed": safe_seed, "responseModalities": ["TEXT", "IMAGE"]}
        payload["generationConfig"]["imageConfig"] = {"imageSize": str(resolution).upper()}
        
        api_aspect_ratio = None
        if aspect_ratio == "原图1比例":
            if first_image_ratio is not None:
                # Snap ratio string for API param
                api_aspect_ratio = self.snap_to_aspect_ratio(first_image_ratio)
        elif aspect_ratio and aspect_ratio != "Free":
            api_aspect_ratio = str(aspect_ratio)

        if api_aspect_ratio:
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = api_aspect_ratio

        print(f"[ComfyUI-shaobkj] Request imageConfig: size={payload['generationConfig']['imageConfig'].get('imageSize')} aspectRatio={payload['generationConfig']['imageConfig'].get('aspectRatio')} raw_ratio={first_image_ratio}")
        print(f"[ComfyUI-shaobkj] Sending request to {url} with model {model}...")
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        task_id = None

        def return_result(img_tensor, raw_text, pil_image=None):
            # No preview image saving, just return result
            ui_info = {"images": []}
            pbar.update_absolute(100)
            return {"ui": ui_info, "result": (img_tensor, raw_text)}

        def format_basic_api_response(status, pil_image=None):
            lines = [
                f"状态: {status}",
                f"模型: {model}",
                f"分辨率: {resolution}",
                f"图片比例: {aspect_ratio}",
                f"seed: {safe_seed}",
            ]
            if task_id:
                lines.append(f"任务ID: {task_id}")
            if pil_image is not None:
                try:
                    w, h = pil_image.size
                    lines.append(f"实际尺寸: {int(w)}x{int(h)}")
                except Exception:
                    pass
            return "\n".join(lines)

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        wait_seconds = int(等待时间)
        submit_timeout = build_submit_timeout(wait_seconds)

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
                img_res = session.get(image_url, verify=False, timeout=60 if timeout_val is None else timeout_val, proxies=proxies, headers=img_headers)
                img_res.raise_for_status()
                image = Image.open(io.BytesIO(img_res.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            except Exception:
                return None

        def extract_image_by_mode(res_json):
            if accept_mode == "智能模式":
                return extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60 if timeout_val is None else timeout_val)
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


        # ---------------------------------------------------------
        # Progress Bar Simulator Logic
        # ---------------------------------------------------------
        progress_state = {"stop": False}

        def progress_simulator():
            current = 0.0
            while not progress_state["stop"] and current < 95.0:
                time.sleep(0.1) # Update frequently for smoothness
                
                # Simulation curve: Fast start -> Slow down -> Crawl
                if current < 30: 
                    step = 2.0  # 0-30%: Very fast (1.5s)
                elif current < 60: 
                    step = 0.5  # 30-60%: Moderate (6s)
                elif current < 80: 
                    step = 0.2  # 60-80%: Slow (10s)
                else: 
                    step = 0.05 # 80-95%: Crawling (indefinite)
                
                current += step
                if current > 95: current = 95
                pbar.update_absolute(int(current))

        # Start progress thread
        import threading
        t_progress = threading.Thread(target=progress_simulator)
        t_progress.daemon = True # Ensure it dies if main thread dies
        t_progress.start()

        def try_openai_fallback():
            # Strict mode: Disable fallback for high resolution requests
            if str(resolution).lower() in ["2k", "4k"]:
                print(f"[ComfyUI-shaobkj] Fallback disabled for {resolution} resolution to prevent quality degradation.")
                return None

            openai_base = base_origin[:-3] if base_origin.endswith("/v1") else base_origin
            openai_url = f"{openai_base}/v1/chat/completions"
            openai_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            openai_content = [{"type": "text", "text": final_prompt}]
            for b64_str in image_b64_list:
                openai_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})
            openai_payload = {
                "model": model,
                "messages": [{"role": "user", "content": openai_content}],
                "temperature": temperature,
                "seed": safe_seed,
            }
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
            try:
                openai_json = openai_resp.json()
            except json.JSONDecodeError:
                return None
            return extract_image_by_mode(openai_json)
        
        def get_task_id_from_headers(resp):
            headers_map = getattr(resp, "headers", {}) or {}
            task_id_local = headers_map.get("X-Task-Id") or headers_map.get("Task-Id") or headers_map.get("task_id") or headers_map.get("task-id")
            if task_id_local:
                return task_id_local
            location = headers_map.get("Location") or headers_map.get("location")
            if isinstance(location, str) and location:
                m = re.search(r"/([^/]+)/?$", location.strip())
                if m:
                    return m.group(1)
            return None

        try:
            response = post_json_with_retry(
                session,
                url,
                headers=headers,
                payload=payload,
                timeout=submit_timeout,
                proxies=proxies,
                verify=False,
            )
            # pbar.update_absolute(50) # Removed static update



            if response.status_code not in (200, 201, 202):
                print(f"[ComfyUI-shaobkj] API Error Status: {response.status_code}")
                try:
                    err_json = response.json()
                    print(f"[ComfyUI-shaobkj] API Error Details: {json.dumps(err_json, indent=2, ensure_ascii=False)}")
                    if "insufficient_user_quota" in str(err_json) or "余额不足" in str(err_json):
                        print("\n[ComfyUI-shaobkj] ⚠️ 警告: 检测到 API 余额不足。")
                        print(f"[ComfyUI-shaobkj] 当前使用的 API Key (末四位): ...{api_key[-4:] if len(api_key) > 4 else api_key}")
                        print("[ComfyUI-shaobkj] 请检查您使用的 API Key 是否与显示余额的账户一致。\n")
                except Exception:
                    print(f"[ComfyUI-shaobkj] API Error Body: {response.text}")

            response.raise_for_status()
            
            try:
                res_json = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                # Handle empty or malformed JSON
                raw_text = response.text
                if not raw_text or not raw_text.strip():
                     print(f"[ComfyUI-shaobkj] Warning: API returned empty response body (HTTP {response.status_code}), try OpenAI fallback")
                     task_id = get_task_id_from_headers(response)
                     if task_id:
                         res_json = {}
                     else:
                         fallback_img = try_openai_fallback()
                         if fallback_img:
                            final_img = fallback_img
                            return return_result(pil_to_tensor(final_img), format_basic_api_response("成功", pil_image=final_img), pil_image=final_img)
                         raise RuntimeError(f"API Error: Empty response body (HTTP {response.status_code})")
                else:
                     print(f"[ComfyUI-shaobkj] JSON Decode Error: {e}")
                     print(f"[ComfyUI-shaobkj] Response Content (first 500 chars): {raw_text[:500]}")
                     fallback_img = try_openai_fallback()
                     if fallback_img:
                        final_img = fallback_img
                        return return_result(pil_to_tensor(final_img), format_basic_api_response("成功", pil_image=final_img), pil_image=final_img)
                     raise RuntimeError(f"Invalid JSON response from API: {e}")

            pbar.update_absolute(70)

            if isinstance(res_json, dict):
                extracted_img = extract_image_by_mode(res_json)
                if extracted_img:
                    w, h = extracted_img.size
                    print(f"[ComfyUI-shaobkj] Response image size: {int(w)}x{int(h)}")
                    final_img = extracted_img
                    return return_result(pil_to_tensor(final_img), format_basic_api_response("成功", pil_image=final_img), pil_image=final_img)

            if isinstance(res_json, dict):
                task_id = res_json.get("id") or res_json.get("task_id")
                if not task_id and "data" in res_json and isinstance(res_json["data"], dict):
                    task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
            if not task_id:
                task_id = get_task_id_from_headers(response)
            if task_id:
                print(f"[ComfyUI-shaobkj] 任务ID: {task_id}. 开始轮询状态...")
                poll_url = f"{url}/{task_id}"
                
                # Use user-provided timeout or default to 86400 (24h) if 0
                user_timeout = int(等待时间)
                poll_timeout_val = 86400 if user_timeout == 0 else user_timeout
                
                start_time = time.time()
                current_p = 70
                fail_count = 0
                poll_attempts = 0
                done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
                failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}

                while True:
                    elapsed = time.time() - start_time
                    remaining = poll_timeout_val - elapsed
                    
                    # Force timeout check
                    if remaining <= 0:
                        raise RuntimeError(f"图像生成超时 ({poll_timeout_val}秒)。如果需要更长等待时间，请增加'等待时间'参数。")

                    # Adaptive polling interval: 
                    # First 5 attempts: 1s (fast check for quick tasks)
                    # Next 10 attempts: 2s
                    # Afterwards: 3s
                    if poll_attempts < 5:
                        sleep_time = 1.0
                    elif poll_attempts < 15:
                        sleep_time = 2.0
                    else:
                        sleep_time = 3.0
                    
                    # Ensure we don't sleep longer than remaining time
                    sleep_time = min(sleep_time, max(0.0, remaining))
                    time.sleep(sleep_time)
                    
                    poll_attempts += 1
                    current_p = min(95, current_p + 1) # Slow down progress bar update
                    pbar.update_absolute(current_p)

                    try:
                        # Calculate request timeout for this specific poll
                        # If infinite wait (0), use 30s per request
                        # If finite wait, use remaining time but clamp to [1, 30]
                        if user_timeout == 0:
                             poll_req_timeout = 30
                        else:
                             poll_req_timeout = max(1, min(30, int(remaining)))

                        poll_resp = session.get(
                            poll_url,
                            headers=headers,
                            params={"_t": int(time.time() * 1000)},
                            verify=False,
                            timeout=poll_req_timeout,
                            proxies=proxies,
                        )
                        fail_count = 0
                    except Exception as e:
                        fail_count += 1
                        if fail_count >= 10:
                            raise RuntimeError(f"Polling failed 10 times consecutively. Last error: {e}")
                        continue

                    if poll_resp.status_code != 200:
                        continue

                    try:
                        poll_json = poll_resp.json()
                    except json.JSONDecodeError:
                        # Handle invalid JSON in polling response (e.g. empty body or html error)
                        # We just continue polling, assuming it's a transient issue
                        # Only print warning if it happens repeatedly or verbose logging is needed
                        # print(f"[ComfyUI-shaobkj] Warning: Polling response invalid JSON (Attempt {poll_attempts})")
                        continue

                    extracted_img = extract_image_by_mode(poll_json)
                    if extracted_img:
                        w, h = extracted_img.size
                        print(f"[ComfyUI-shaobkj] Response image size (poll): {int(w)}x{int(h)}")
                        final_img = extracted_img
                        return return_result(pil_to_tensor(final_img), format_basic_api_response("成功", pil_image=final_img), pil_image=final_img)

                    status = None
                    if isinstance(poll_json, dict):
                        status = poll_json.get("status") or poll_json.get("task_status")
                        if not status and "data" in poll_json and isinstance(poll_json["data"], dict):
                            status = poll_json["data"].get("status") or poll_json["data"].get("task_status")
                    status_str = str(status).strip().upper() if status is not None else ""
                    if status_str in failed_statuses:
                        raise RuntimeError(f"图像生成失败: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")
                    if status_str in done_statuses:
                        raise RuntimeError(f"任务已完成但未找到图像: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")

            fallback_img = try_openai_fallback()
            if fallback_img:
                    final_img = fallback_img
                    return return_result(pil_to_tensor(final_img), format_basic_api_response("成功", pil_image=final_img), pil_image=final_img)
            raise RuntimeError(f"No image found in API response. Response: {sanitize_text(json.dumps(res_json, ensure_ascii=False))}")
        except Exception as e:
            # Stop progress thread on error
            progress_state["stop"] = True
            t_progress.join(timeout=1.0)
            
            error_msg = str(e)
            if "504" in error_msg:
                raise RuntimeError("请求超时 (504 Gateway Time-out)。服务器处理时间过长，请稍后重试。")
            if "Total execution time exceeded limit" in error_msg:
                raise RuntimeError(f"等待超时 ({int(等待时间)}秒)。任务执行时间超过了设定的'等待时间'，已被强制终止。")
            if "Read timed out" in error_msg or "Connect timed out" in error_msg:
                 raise RuntimeError(f"网络连接超时。网络响应慢，或者您设定的等待时间 ({int(等待时间)}秒) 不足以完成任务。请检查网络或增加等待时间。")
            if "Expecting value: line 1 column 1" in error_msg:
                raise RuntimeError(f"请求失败 (数据不完整)。服务器连接不稳定，接收到的数据不完整。请增加等待时间或检查网络。")
            raise RuntimeError(f"请求失败: {error_msg}")
            
        finally:
            # Always stop progress thread and set to 100% on finish
            progress_state["stop"] = True
            if t_progress.is_alive():
                t_progress.join(timeout=1.0)
            pbar.update_absolute(100)


# ----------------------------------------------------------------------------
# Background Worker for Text-to-Image Batch
# ----------------------------------------------------------------------------

def snap_to_aspect_ratio(ratio):
    """
    Snaps a float ratio (width/height) to the nearest standard aspect ratio string.
    """
    standards = {
        1.0: "1:1",
        4/3: "4:3",
        3/4: "3:4",
        3/2: "3:2",
        2/3: "2:3",
        16/9: "16:9",
        9/16: "9:16",
        21/9: "21:9",
        9/21: "9:21"
    }
    
    closest_dist = float('inf')
    closest_str = "1:1"
    
    for r_val, r_str in standards.items():
        dist = abs(ratio - r_val)
        if dist < closest_dist:
            closest_dist = dist
            closest_str = r_str
            
    return closest_str

def run_batch_generation_task(data):
    # Use provided task_id or generate new one
    task_id_local = data.get("task_id")
    if not task_id_local:
        task_id_local = f"task_{int(time.time())}_{random.randint(1000,9999)}"
    
    print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Starting task {task_id_local}...")
    
    # Initial status update
    update_async_task(task_id_local, {
        "status": "running",
        "submitted_at": int(time.time()),
        "prompt": data.get("prompt", "")[:50] + "...",
        "type": "gen_batch"
    })

    try:
        # Parse params
        api_key = data.get("api_key")
        api_url_base = data.get("api_url", "https://yhmx.work")
        model = data.get("model", "gemini-3-pro-image-preview")
        use_proxy = data.get("use_proxy", False)
        resolution = data.get("resolution", "1k")
        prompt = data.get("prompt", "")
        aspect_ratio = data.get("aspect_ratio", "Free")
        wait_time = int(data.get("wait_time", 0))
        seed_val = int(data.get("seed", 0))
        save_path_input = data.get("save_path", "")
        save_format_input = data.get("save_format", "JPEG (默认95%)")
        accept_mode = data.get("accept_mode", "智能模式")
        subject_text = data.get("subject_text", "")

        if not api_key:
             raise ValueError("API Key is required")

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
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key, "Authorization": f"Bearer {api_key}"}
        
        # Force generation instruction
        final_prompt = str(prompt) + "\n\n(Generate an image based on this description)"
        parts = [{"text": final_prompt}]
        
        # Handle Images (New)
        tensor_images = data.get("tensor_images", [])
        long_side = int(data.get("long_side", 1280))
        
        first_image_ratio = None
        subject_crop_ratio = None
        if isinstance(subject_text, str) and subject_text.strip():
            if aspect_ratio == "原图1比例" and tensor_images:
                first_img = tensor_images[0]
                first_pil = tensor_to_pil(first_img) if isinstance(first_img, torch.Tensor) else first_img
                if first_pil is not None:
                    w0, h0 = first_pil.size
                    if h0 > 0:
                        subject_crop_ratio = snap_to_aspect_ratio(float(w0) / float(h0))
            elif aspect_ratio and aspect_ratio != "Free":
                subject_crop_ratio = str(aspect_ratio)

        for idx, img in enumerate(tensor_images):
            try:
                pil_img = tensor_to_pil(img) if isinstance(img, torch.Tensor) else img
                use_png = "PNG" in str(save_format_input)
                pil_img = resize_pil_long_side(pil_img, long_side)
                if subject_crop_ratio:
                    detect_timeout = 30 if wait_time <= 0 else max(5, min(30, int(wait_time)))
                    bbox = detect_subject_bbox(pil_img, subject_text.strip(), base_origin, api_key, use_proxy, detect_timeout)
                    pil_img = crop_image_to_ratio(pil_img, subject_crop_ratio, bbox)
                if pil_img is None:
                    continue
                b64_str = encode_pil_image(pil_img, use_png, quality=95)
                img_ratio = float(pil_img.size[0]) / float(pil_img.size[1])
                if idx == 0:
                    first_image_ratio = img_ratio
                
                if b64_str:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png" if use_png else "image/jpeg",
                            "data": b64_str
                        }
                    })
            except Exception as e:
                print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Error encoding image: {e}")
        
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
                "responseModalities": ["IMAGE"]
            }
        }
        payload["generationConfig"]["imageConfig"] = {"imageSize": str(resolution).upper()}

        target_aspect_ratio = aspect_ratio
        
        if target_aspect_ratio == "原图1比例":
            if first_image_ratio is not None:
                target_aspect_ratio = snap_to_aspect_ratio(first_image_ratio)
            else:
                # Fallback if no image provided but "原图1比例" selected? 
                # Maybe default to 1:1 or keep as is (which API might ignore or error)
                # Let's default to "1:1" if no image
                target_aspect_ratio = "1:1"

        if target_aspect_ratio and target_aspect_ratio != "Free" and target_aspect_ratio != "原图1比例":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(target_aspect_ratio)

        print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Request imageConfig: size={payload['generationConfig']['imageConfig'].get('imageSize')} aspectRatio={payload['generationConfig']['imageConfig'].get('aspectRatio')} raw_ratio={first_image_ratio}")
        # Send Request
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(use_proxy))
        submit_timeout = build_submit_timeout(wait_time)
        
        # Helper Functions
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
            print(f"[ComfyUI-shaobkj] [Concurrent-Batch] {task_id_local}: API Error Status: {response.status_code}")
            try:
                print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Error Body: {response.text[:200]}")
            except Exception:
                pass
        
        response.raise_for_status()
        
        try:
            res_json = response.json()
        except (json.JSONDecodeError, ValueError):
             raw_text = response.text
             if not raw_text or not raw_text.strip():
                 print(f"[ComfyUI-shaobkj] [Concurrent-Batch] {task_id_local}: Warning: Empty response body")
                 task_id = get_task_id_from_headers(response)
                 if task_id:
                     res_json = {"id": task_id}
                 else:
                     raise RuntimeError(f"API Error: Empty response body (HTTP {response.status_code})")
             else:
                 raise

        # Verify response content
        extracted_img = None
        if not res_json or (not res_json.get("candidates") and not res_json.get("id") and not res_json.get("name") and not res_json.get("data")):
            print(f"[ComfyUI-shaobkj] {task_id_local}: Warning - Empty or invalid response from API.")

        # Extract Result
        extracted_img = extract_image_by_mode(res_json)
        
        remote_task_id = None
        
        if not extracted_img:
             remote_task_id = res_json.get("id") or res_json.get("task_id")
             if not remote_task_id and "name" in res_json:
                 remote_task_id = res_json["name"]
             if not remote_task_id and "data" in res_json:
                 remote_task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
             if not remote_task_id:
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
                         print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Polling error: {e}")
                         if fail_count > 10:
                             raise

        # Save Result
        if extracted_img:
            w, h = extracted_img.size
            print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Response image size: {int(w)}x{int(h)}")
            final_img = extracted_img
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
                filename = f"batch_gen_{int(time.time())}_{random.randint(1000,9999)}{ext}"
            
            # Determine output directory
            out_dir = folder_paths.get_output_directory()
            if save_path_input and isinstance(save_path_input, str) and save_path_input.strip():
                custom_dir = save_path_input.strip()
                if not os.path.isabs(custom_dir):
                    custom_dir = os.path.join(out_dir, custom_dir)
                try:
                    os.makedirs(custom_dir, exist_ok=True)
                    out_dir = custom_dir
                except Exception as e:
                    print(f"[ComfyUI-shaobkj] [Concurrent-Batch] Failed to create custom dir {custom_dir}: {e}")

            out_path = os.path.join(out_dir, filename)
            
            # Check for overwrite
            counter = 1
            original_base_name = os.path.splitext(filename)[0]
            while os.path.exists(out_path):
                new_filename = f"{original_base_name}_{counter}{ext}"
                out_path = os.path.join(out_dir, new_filename)
                filename = new_filename
                counter += 1
            
            final_img.save(out_path, **save_params)
            
            print(f"[ComfyUI-shaobkj] [Concurrent-Batch] {task_id_local}: Success! Saved to {out_path}")
            
            update_async_task(task_id_local, {
                "status": "success",
                "image_path": out_path,
                "completed_at": int(time.time())
            })
            
            save_local_record("Concurrent_Batch", str(remote_task_id or task_id_local), "Success", api_url_base)
            PromptServer.instance.send_sync("shaobkj.concurrent.success", {"task_id": task_id_local, "filename": filename, "path": out_path})
            
            return task_id_local
        else:
            brief = sanitize_text(json.dumps(res_json, ensure_ascii=False))
            raise RuntimeError(f"No image data found in response: {brief}")

    except Exception as e:
        err_msg = str(e)
        print(f"[ComfyUI-shaobkj] [Concurrent-Batch] {task_id_local}: {err_msg}")
        update_async_task(task_id_local, {
            "status": "failed",
            "error": err_msg,
            "completed_at": int(time.time())
        })
        raise RuntimeError(err_msg)


class Shaobkj_APINode_Batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "生成内容描述，支持多行；推荐：每行一条提示词"}),
                "提示词列表": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "是否将输入视为提示词列表；开启时按行拆分，关闭时视为单条提示词"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                        "智能加载",
                    ],
                    {"default": "gemini-3-pro-image-preview", "tooltip": "模型选择或智能加载；推荐：gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k", "tooltip": "输出分辨率档位；推荐：1k"}),
                "图片比例": (
                    ["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"],
                    {"default": "原图1比例", "tooltip": "输出画面比例；推荐：原图1比例"},
                ),
                "接收模式": (["智能模式", "URL", "B64"], {"default": "智能模式", "tooltip": "API 返回内容处理方式；推荐：智能模式"}),
                "主体文本": ("STRING", {"default": "", "multiline": False, "tooltip": "主体识别裁切关键词；推荐：留空"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "出图数量": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "tooltip": "单次提交的任务总数/循环次数；推荐：1"}),
                "指定文件名": ("STRING", {"default": "", "multiline": False, "placeholder": "为空则自动命名，输入则自动添加序号", "tooltip": "为空自动命名，可自定义前缀；推荐：留空"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "Batch拆分模式": ("BOOLEAN", {"default": True, "tooltip": "是否拆分批次提交；推荐：开启"}),
                "Batch对齐方式": (["循环补全(Max)", "裁切对齐(Min)"], {"default": "循环补全(Max)", "tooltip": "批次对齐策略；推荐：循环补全(Max)"}),
                "保存路径": ("STRING", {"default": "Shaobkj_Concurrent", "multiline": False, "tooltip": "相对输出目录的子路径；推荐：Shaobkj_Concurrent"}),
                "保存格式": (["JPEG (默认95%)", "PNG (无损)", "WEBP (无损)"], {"default": "JPEG (默认95%)", "tooltip": "输出保存格式；推荐：JPEG (默认95%)"}),
                "最大并发数": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "后台最大同时执行任务数；推荐：5"}),
                "并发间隔": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "批量任务提交间隔(秒)；推荐：1.0"}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("任务ID列表", "提交状态")
    FUNCTION = "generate_images_batch"
    CATEGORY = "🤖shaobkj-APIbox"
    OUTPUT_NODE = True

    def generate_images_batch(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 接收模式, 主体文本, 输入图像_长边设置=1280, 出图数量=1, 指定文件名="", seed=0, Batch拆分模式=True, Batch对齐方式="循环补全(Max)", 保存路径="Shaobkj_Concurrent", 保存格式="JPEG (默认95%)", 最大并发数=5, 并发间隔=1.0, 提示词列表=False, **kwargs):
        # Unwrap parameters because INPUT_IS_LIST = True
        def get_val(v, default=None):
            if isinstance(v, list) and len(v) > 0:
                return v[0]
            return v
        
        # Helper to normalize generic list inputs (recursively flatten)
        def normalize_list_input(val):
            flat_list = []
            if isinstance(val, list):
                for item in val:
                    flat_list.extend(normalize_list_input(item))
            else:
                flat_list.append(val)
            return flat_list

        # Helper to normalize image inputs (Tensor to PIL)
        def normalize_image_input(val):
            flat_list = []
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, list):
                         flat_list.extend(normalize_image_input(item))
                    elif isinstance(item, torch.Tensor):
                        # item is [B,H,W,C]
                        for i in range(item.shape[0]):
                            flat_list.append(Image.fromarray(np.clip(255. * item[i].cpu().numpy(), 0, 255).astype(np.uint8)))
            elif isinstance(val, torch.Tensor):
                 for i in range(val.shape[0]):
                    flat_list.append(Image.fromarray(np.clip(255. * val[i].cpu().numpy(), 0, 255).astype(np.uint8)))
            return flat_list
        
        api_key_val = get_val(API密钥)
        api_url_val = get_val(API地址)
        model_val = get_val(模型选择)
        use_proxy_val = get_val(使用系统代理)
        
        long_side_val = int(get_val(输入图像_长边设置))
        batch_count_val = int(get_val(出图数量))
        
        # Lists
        resolution_list = normalize_list_input(分辨率)
        aspect_ratio_list = normalize_list_input(图片比例)
        subject_text_list = normalize_list_input(主体文本)
        seed_list = normalize_list_input(seed)
        accept_mode_list = normalize_list_input(接收模式)
        save_path_list = normalize_list_input(保存路径)
        save_format_list = normalize_list_input(保存格式)
        filename_prefix_list = normalize_list_input(指定文件名)
        
        # Process Image Inputs
        normalized_images = {}
        for k, v in kwargs.items():
            if k.startswith("image_"):
                normalized_images[k] = normalize_image_input(v)
        sorted_image_keys = sorted(normalized_images.keys())
        
        # Config
        batch_split_val = get_val(Batch拆分模式, True)
        batch_align_val = get_val(Batch对齐方式, "循环补全(Max)")
        submit_interval_val = float(get_val(并发间隔, 1.0))
        max_workers_val = int(get_val(最大并发数, 5))

        if not api_key_val:
            raise ValueError("API Key is required.")

        # Prepare Prompts
        prompt_list_mode = get_val(提示词列表, False)
        prompts = []
        raw_prompts = normalize_list_input(提示词)
        
        if prompt_list_mode:
            # List Mode: Split lines, ignore batch_count logic (effectively 1 per line)
            for p in raw_prompts:
                if isinstance(p, str):
                    lines = [line.strip() for line in p.splitlines() if line.strip()]
                    prompts.extend(lines)
        else:
            # Single Prompt Mode: Keep multiline string as one prompt, use batch_count
            for p in raw_prompts:
                if isinstance(p, str) and p.strip():
                    # Repeat prompt for batch_count times
                    for _ in range(batch_count_val):
                        prompts.append(p)

        prompts = [str(p) for p in prompts if str(p).strip()]
        if not prompts:
             print("[ComfyUI-shaobkj] ⚠️ 提示词为空，跳过本次生成。等待上游节点输入...")
             return ([], [])

        # ---------------------------------------------------------------------------
        # Feature: Auto-disable 'Image Count' if Prompt List is provided
        # If we have multiple prompts (from list input or multiline split),
        # we strictly follow the prompt count (1 image per prompt).
        # ---------------------------------------------------------------------------
        if prompt_list_mode and batch_count_val > 1:
            print(f"[ComfyUI-shaobkj] ℹ️ 提示词列表模式已开启 (检测到 {len(prompts)} 条提示词)。'出图数量'参数 ({batch_count_val}) 将失效，强制按提示词列表生成。")
            batch_count_val = 1
        # ---------------------------------------------------------------------------

        # Prepare Tasks
        task_list = []
        base_task_id = f"batch_{int(time.time())}_{random.randint(1000,9999)}"
        
        # Determine Batch Size
        if not batch_split_val:
             final_batch_size = 1
        else:
             lengths = []
             lengths.append(len(prompts))
             if len(seed_list) > 1: lengths.append(len(seed_list))
             if len(resolution_list) > 1: lengths.append(len(resolution_list))
             if len(aspect_ratio_list) > 1: lengths.append(len(aspect_ratio_list))
             if len(subject_text_list) > 1: lengths.append(len(subject_text_list))
             
             # Add Image Lengths
             for k, v in normalized_images.items():
                 if len(v) > 1: lengths.append(len(v))
             
             # Add Batch Count
             if batch_count_val > 1:
                 lengths.append(batch_count_val)
             
             if not lengths: 
                 final_batch_size = 1
             elif batch_align_val == "裁切对齐(Min)":
                 final_batch_size = min(lengths)
             else:
                 final_batch_size = max(lengths)

        print(f"[Shaobkj-Batch] Final Batch Size: {final_batch_size} (Mode: {batch_align_val}, Count: {batch_count_val})")
        
        for i in range(final_batch_size):
            sub_task_id = f"{base_task_id}_{i}"
            p = prompts[i % len(prompts)]
            
            if len(seed_list) == 1:
                s_val = int(seed_list[0]) + i
            else:
                s_val = int(seed_list[i % len(seed_list)])

            # Collect Task Images
            task_imgs = []
            for k in sorted_image_keys:
                imgs = normalized_images[k]
                if imgs:
                    task_imgs.append(imgs[i % len(imgs)])

            fn_prefix = filename_prefix_list[i % len(filename_prefix_list)]
            out_fn = None
            if fn_prefix and str(fn_prefix).strip():
                out_fn = f"{str(fn_prefix).strip()}_{i+1}"

            task_data = {
                "task_id": sub_task_id,
                "api_key": api_key_val,
                "api_url": api_url_val,
                "model": model_val,
                "use_proxy": use_proxy_val,
                "resolution": resolution_list[i % len(resolution_list)],
                "prompt": p,
                "aspect_ratio": aspect_ratio_list[i % len(aspect_ratio_list)],
                "accept_mode": accept_mode_list[i % len(accept_mode_list)] if accept_mode_list else "智能模式",
                "subject_text": subject_text_list[i % len(subject_text_list)] if subject_text_list else "",
                "wait_time": 0,
                "seed": s_val,
                "save_path": save_path_list[i % len(save_path_list)],
                "save_format": save_format_list[i % len(save_format_list)],
                "output_filename": out_fn,
                "long_side": long_side_val,
                "tensor_images": task_imgs
            }
            task_list.append(task_data)

        # Background Monitor
        def background_batch_monitor(tasks, max_workers, split_mode, interval):
            total_tasks = len(tasks)
            print(f"[ComfyUI-shaobkj-BG] [Concurrent-Batch] Monitor started for {total_tasks} tasks. (Workers: {max_workers})")
            
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
                        status_str = f"失败: {error_msg}"
                    
                    print(f"[ComfyUI-shaobkj-BG] [Concurrent-Batch] {completed_count}/{total_tasks} | 成功: {success_count} | 失败: {fail_count} | {tid} {status_str}")

                    if completed_count == total_tasks:
                        summary = f"[ComfyUI-shaobkj-BG] [Concurrent-Batch] 完成。总: {total_tasks} | 成功: {success_count} | 失败: {fail_count}"
                        if fail_count > 0: summary += f" | 详情: {'; '.join(failure_reasons)}"
                        print(summary)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i, task_data in enumerate(tasks):
                    if split_mode and interval > 0 and i > 0:
                        time.sleep(interval)
                    
                    future = executor.submit(run_batch_generation_task, task_data)
                    future.add_done_callback(lambda f, t=task_data["task_id"]: on_task_done(f, t))

        # Launch Thread
        max_workers = max_workers_val if max_workers_val > 0 else 1000
        t = threading.Thread(target=background_batch_monitor, args=(task_list, max_workers, batch_split_val, submit_interval_val))
        t.daemon = True
        t.start()
        
        msg = f"已提交 {len(task_list)} 个生成任务到后台。"
        print(f"[ComfyUI-shaobkj] {msg}")
        
        generated_ids = [t["task_id"] for t in task_list]
        status_list = ["Submitted" for _ in task_list]
        
        return (generated_ids, status_list)
