import os
import json
import torch
import numpy as np
from PIL import Image
import io
import folder_paths
import base64
import re
import random
import time
import traceback
import concurrent.futures
from urllib.parse import urlparse
import pandas as pd

import torch.nn.functional as F

from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    pil_to_tensor,
    post_json_with_retry,
    extract_image_from_json,
    smart_pad_images_to_tensor,
)
from comfy.utils import ProgressBar


def sanitize_text(s, max_len=1200):
    t = "" if s is None else str(s)
    t = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[省略]", t)
    t = re.sub(r"[A-Za-z0-9+/=]{200,}", "[省略]", t)
    if len(t) > max_len:
        t = t[:max_len] + "...(省略)"
    return t


class Shaobkj_APINode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                    ],
                    {"default": "gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k"}),
                "图片比例": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"],
                    {"default": "原图1比例"},
                ),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
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

    def get_target_size(self, resolution, aspect_ratio, first_image_ratio=None):
        target_map = {"1k": 1024, "2k": 2048, "4k": 4096}
        target = target_map.get(str(resolution).lower(), 1024)

        ar = str(aspect_ratio)
        
        # Handle "原图1比例" (Use first image ratio if available)
        if ar == "原图1比例" and first_image_ratio is not None:
             # Snap the actual float ratio to nearest standard string
             snapped_ar_str = self.snap_to_aspect_ratio(first_image_ratio)
             ar = snapped_ar_str # e.g., "16:9"

        if ar == "原图1比例" or ar == "Free": 
             return target, target
             
        if ":" in ar:
            try:
                a, b = ar.split(":", 1)
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

    def resize_and_encode_image(self, image_tensor, long_side):
        if image_tensor is None:
            return None, 1.0
        
        # Convert tensor [B,H,W,C] to PIL
        i = 255. * image_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])
        
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        
        # Calculate new size maintaining aspect ratio
        if original_width > original_height:
            new_width = long_side
            new_height = int(long_side / aspect_ratio)
        else:
            new_height = long_side
            new_width = int(long_side * aspect_ratio)
            
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffered = io.BytesIO()
        img_resized.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str, aspect_ratio

    def generate_image(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词, 图片比例, 等待时间, seed, **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        resolution = 分辨率
        prompt = 提示词
        aspect_ratio = 图片比例
        long_side_limit = int(kwargs.get("输入图像-长边设置", 1280))
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        seed_value = seed

        temperature = 0.7

        if not api_key:
            raise ValueError("API Key is required.")

        model = 模型选择

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        url = f"{base_origin}/v1beta/models/{model}:generateContent"

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
        
        for idx, (name, tensor) in enumerate(image_inputs):
            b64_str, ratio = self.resize_and_encode_image(tensor, long_side_limit)
            if idx == 0:
                first_image_ratio = ratio
            if b64_str:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
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
        submit_timeout = build_submit_timeout(int(等待时间))

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
            res_json = response.json()
            pbar.update_absolute(70)

            extracted_img = extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60 if timeout_val is None else timeout_val)
            if extracted_img:
                return return_result(pil_to_tensor(extracted_img), format_basic_api_response("成功", pil_image=extracted_img), pil_image=extracted_img)

            if isinstance(res_json, dict):
                task_id = res_json.get("id") or res_json.get("task_id")
                if not task_id and "data" in res_json and isinstance(res_json["data"], dict):
                    task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
            if task_id:
                print(f"[ComfyUI-shaobkj] 任务ID: {task_id}. 开始轮询状态...")
                poll_url = f"{url}/{task_id}"
                
                # Use user-provided timeout or default to 86400 (24h) if 0
                user_timeout = int(等待时间)
                poll_timeout_val = 86400 if user_timeout == 0 else user_timeout
                
                start_time = time.time()
                current_p = 70
                fail_count = 0
                done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
                failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}

                while True:
                    elapsed = time.time() - start_time
                    remaining = poll_timeout_val - elapsed
                    
                    # Force timeout check
                    if remaining <= 0:
                        raise RuntimeError(f"图像生成超时 ({poll_timeout_val}秒)。如果需要更长等待时间，请增加'等待时间'参数。")

                    time.sleep(min(5, max(0.0, remaining)))
                    current_p = min(95, current_p + 2)
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

                    poll_json = poll_resp.json()
                    extracted_img = extract_image_from_json(poll_json, session, proxies, api_key, api_origin, timeout_val=60 if timeout_val is None else timeout_val)
                    if extracted_img:
                        return return_result(pil_to_tensor(extracted_img), format_basic_api_response("成功", pil_image=extracted_img), pil_image=extracted_img)

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

            raise RuntimeError(f"No image found in API response. Response: {sanitize_text(json.dumps(res_json, ensure_ascii=False))}")
        except Exception as e:
            # Stop progress thread on error
            progress_state["stop"] = True
            t_progress.join(timeout=1.0)
            
            error_msg = str(e)
            if "504" in error_msg:
                raise RuntimeError("请求超时 (504 Gateway Time-out)。服务器处理时间过长，请稍后重试。")
            raise RuntimeError(f"请求失败: {error_msg}")
            
        finally:
            # Always stop progress thread and set to 100% on finish
            progress_state["stop"] = True
            if t_progress.is_alive():
                t_progress.join(timeout=1.0)
            pbar.update_absolute(100)


class Shaobkj_APINode_Batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词列表": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "一只猫\n一只狗", "placeholder": "每行一个提示词，或者拖入CSV/Excel文件路径"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型选择": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k"}),
                "图片比例": (
                    ["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],
                    {"default": "Free"},
                ),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "并发数": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "0=智能并发（按任务数自动扩展，上限10）"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "文件列名": ("STRING", {"default": "prompt", "multiline": False, "tooltip": "CSV/Excel中提示词所在的列名"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "API响应")
    FUNCTION = "generate_images_batch"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_images_batch(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词列表, 图片比例, 等待时间, 并发数, seed, 文件列名="prompt", **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        model = 模型选择
        resolution = 分辨率
        aspect_ratio = 图片比例
        timeout_val = None if int(等待时间) == 0 else int(等待时间)

        if not api_key:
            raise ValueError("API Key is required.")

        # --- Phase 1 Feature: Excel/CSV Support ---
        raw_input = str(提示词列表 or "").strip()
        prompts = []
        
        # Check if input looks like a file path and exists
        if raw_input and (raw_input.endswith('.csv') or raw_input.endswith('.xlsx') or raw_input.endswith('.xls')) and os.path.exists(raw_input.strip('"')):
            file_path = raw_input.strip('"')
            print(f"[Shaobkj-Batch] Reading prompts from file: {file_path}")
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                if 文件列名 in df.columns:
                    prompts = df[文件列名].dropna().astype(str).tolist()
                    print(f"[Shaobkj-Batch] Loaded {len(prompts)} prompts from column '{文件列名}'")
                else:
                    print(f"[Shaobkj-Batch] Warning: Column '{文件列名}' not found. Columns: {df.columns.tolist()}")
                    raise ValueError(f"列 '{文件列名}' 不存在于文件中")
            except Exception as e:
                raise ValueError(f"读取文件失败: {e}")
        else:
            # Fallback to standard multiline text
            prompts = [p.strip() for p in raw_input.splitlines() if p.strip()]

        if not prompts:
            raise ValueError("提示词列表不能为空。")

        disable_insecure_request_warnings()

        base_headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        submit_timeout = build_submit_timeout(int(等待时间))
        session, proxies = create_requests_session(bool(使用系统代理))

        # ---------------------------------------------------------
        # Progress Bar Simulator Logic (Batch)
        # ---------------------------------------------------------
        pbar = ProgressBar(100)
        pbar.update_absolute(0)
        progress_state = {"stop": False}

        def progress_simulator():
            current = 0.0
            while not progress_state["stop"] and current < 95.0:
                time.sleep(0.1)
                if current < 30: step = 2.0
                elif current < 60: step = 0.5
                elif current < 80: step = 0.2
                else: step = 0.05
                current += step
                if current > 95: current = 95
                pbar.update_absolute(int(current))

        import threading
        t_progress = threading.Thread(target=progress_simulator)
        t_progress.daemon = True
        t_progress.start()

        def format_basic_api_response(status, safe_seed, pil_image=None, task_id=None):
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

        def normalize_seed(seed_value):
            safe_seed = int(seed_value)
            if safe_seed < 0:
                safe_seed = random.randint(0, 2147483647)
            if safe_seed > 2147483647:
                safe_seed = safe_seed % 2147483647
            return safe_seed

        def extract_brief_message(obj):
            if isinstance(obj, dict):
                err = obj.get("error")
                if isinstance(err, dict):
                    return err.get("message") or err.get("msg") or err.get("code")
                data = obj.get("data")
                if isinstance(data, dict):
                    err2 = data.get("error")
                    if isinstance(err2, dict):
                        return err2.get("message") or err2.get("msg") or err2.get("code")
                return obj.get("message") or obj.get("msg") or obj.get("error_message") or obj.get("detail")
            return None

        def sanitize_text(s, max_len=600):
            t = "" if s is None else str(s)
            t = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[省略]", t)
            t = re.sub(r"[A-Za-z0-9+/=]{200,}", "[省略]", t)
            if len(t) > max_len:
                t = t[:max_len] + "...(省略)"
            return t

        def generate_one(index, prompt):
            local_seed = normalize_seed(int(seed) + int(index))
            # Session and proxies reused from outer scope
            # Force image generation by appending instruction
            final_prompt = str(prompt) + "\n\n(Generate an image based on this description)"
            parts = [{"text": final_prompt}]
            payload = {"contents": [{"role": "user", "parts": parts}]}
            payload["generationConfig"] = {"temperature": 0.7, "seed": local_seed, "responseModalities": ["TEXT", "IMAGE"]}
            payload["generationConfig"]["imageConfig"] = {"imageSize": str(resolution).upper()}
            if aspect_ratio and aspect_ratio != "Free":
                payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(aspect_ratio)
            task_id = None

            response = post_json_with_retry(
                session,
                url,
                headers=base_headers,
                payload=payload,
                timeout=submit_timeout,
                proxies=proxies,
                verify=False,
            )
            response.raise_for_status()
            res_json = response.json()

            extracted_img = extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60 if timeout_val is None else timeout_val)
            if extracted_img:
                return (pil_to_tensor(extracted_img), format_basic_api_response("成功", local_seed, pil_image=extracted_img, task_id=task_id))

            if isinstance(res_json, dict):
                task_id = res_json.get("id") or res_json.get("task_id")
                if not task_id and "data" in res_json and isinstance(res_json["data"], dict):
                    task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
            if not task_id:
                brief = extract_brief_message(res_json)
                if brief:
                    raise RuntimeError(f"未找到任务ID，API响应: {sanitize_text(brief)}")
                raise RuntimeError(f"未找到任务ID，API响应: {sanitize_text(json.dumps(res_json, ensure_ascii=False))}")

            poll_url = f"{url}/{task_id}"
            poll_timeout_val = 86400 if int(等待时间) == 0 else int(等待时间)
            start_time = time.time()
            fail_count = 0
            done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
            failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}
            while True:
                elapsed = time.time() - start_time
                remaining = poll_timeout_val - elapsed
                if remaining <= 0:
                    raise RuntimeError(f"图像生成超时 ({poll_timeout_val}秒)")
                time.sleep(min(5, max(0.0, remaining)))
                try:
                    poll_req_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))
                    poll_resp = session.get(
                        poll_url,
                        headers=base_headers,
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
                poll_json = poll_resp.json()
                extracted_img = extract_image_from_json(poll_json, session, proxies, api_key, api_origin, timeout_val=60 if timeout_val is None else timeout_val)
                if extracted_img:
                    return (pil_to_tensor(extracted_img), format_basic_api_response("成功", local_seed, pil_image=extracted_img, task_id=task_id))
                
                status = None
                if isinstance(poll_json, dict):
                    status = poll_json.get("status") or poll_json.get("task_status")
                    if not status and "data" in poll_json and isinstance(poll_json["data"], dict):
                        status = poll_json["data"].get("status") or poll_json["data"].get("task_status")
                status_str = str(status).strip().upper() if status is not None else ""
                if status_str in failed_statuses:
                    brief = extract_brief_message(poll_json)
                    if brief:
                        raise RuntimeError(f"图像生成失败: {sanitize_text(brief)}")
                    raise RuntimeError(f"图像生成失败: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")
                if status_str in done_statuses:
                    brief = extract_brief_message(poll_json)
                    if brief:
                        raise RuntimeError(f"任务已完成但未找到图像: {sanitize_text(brief)}")
                    raise RuntimeError(f"任务已完成但未找到图像: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")

        errors = []
        results = []
        concurrency_limit = int(并发数)
        if concurrency_limit <= 0:
            max_workers = min(10, max(1, len(prompts)))
        else:
            max_workers = min(10, max(1, min(concurrency_limit, len(prompts))))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(generate_one, idx, p): (idx, p) for idx, p in enumerate(prompts)}
                for fut in concurrent.futures.as_completed(future_map):
                    idx, p = future_map[fut]
                    try:
                        img_tensor, resp_text = fut.result()
                        results.append((idx, img_tensor, resp_text))
                    except Exception as e:
                        errors.append((idx, sanitize_text(str(e))))
        finally:
            progress_state["stop"] = True
            if t_progress.is_alive():
                t_progress.join(timeout=1.0)
            pbar.update_absolute(100)

        results.sort(key=lambda x: x[0])
        ok_tensors = [r[1] for r in results if isinstance(r[1], torch.Tensor)]
        if not ok_tensors:
            raise RuntimeError(f"批量生成全部失败，示例错误: {errors[0][1] if errors else '未知错误'}")

        # --- Phase 1 Feature: Smart Padding ---
        # The previous pad_tensor_to logic was basic. Now we use the logic inspired by nkxx but adapted for tensors.
        # However, nkxx logic converts PIL->Tensor with padding. Here we already have tensors.
        # The existing logic here already pads to max_h/max_w.
        # Let's verify if we need to change it. 
        # The existing logic:
        # max_h = max(int(t.shape[1]) for t in ok_tensors)
        # max_w = max(int(t.shape[2]) for t in ok_tensors)
        # padded = [pad_tensor_to(t, max_h, max_w) for t in ok_tensors]
        # This is already what "Smart Padding" does (Auto-Padding).
        # So I just need to make sure pad_tensor_to is robust.
        
        def pad_tensor_to_v2(t, max_h, max_w):
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                return t
            b, h, w, c = t.shape
            if h == max_h and w == max_w:
                return t
            # T is [B, H, W, C]
            # Permute to [B, C, H, W] for padding
            tch = t.permute(0, 3, 1, 2)
            pad_w = max_w - w
            pad_h = max_h - h
            # Pad right and bottom
            padded = F.pad(tch, (0, pad_w, 0, pad_h), "constant", 0)
            # Permute back
            return padded.permute(0, 2, 3, 1)

        max_h = max(int(t.shape[1]) for t in ok_tensors)
        max_w = max(int(t.shape[2]) for t in ok_tensors)
        padded = [pad_tensor_to_v2(t, max_h, max_w) for t in ok_tensors]
        batch_tensor = torch.cat(padded, dim=0)

        lines = [f"批量生成完成 | 总数: {len(prompts)} | 成功: {len(ok_tensors)} | 失败: {len(errors)}"]
        if errors:
            for idx, msg in errors[:5]:
                snippet = prompts[idx][:30] if idx < len(prompts) else str(idx)
                lines.append(f"失败[{idx}] {snippet}: {sanitize_text(msg)}")
        api_text = "\n".join(lines)
        return {"ui": {"string": [api_text]}, "result": (batch_tensor, api_text)}
