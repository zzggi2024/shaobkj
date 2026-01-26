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
from urllib.parse import urlparse
import folder_paths
from PIL import Image, ImageOps
from server import PromptServer
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor, as_completed
from comfy.utils import ProgressBar

from .shaobkj_shared import (
    get_config_value,
    create_requests_session,
    disable_insecure_request_warnings,
    build_submit_timeout,
    post_json_with_retry,
    auth_headers_for_same_origin,
    resize_and_encode_image,
    extract_image_from_json,
    save_local_record,
    sanitize_text,
    update_async_task,
    get_all_async_tasks,
    pil_to_tensor
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
                # Use shared helper
                b64_str, img_ratio = resize_and_encode_image(img, long_side)
                if b64_str:
                    image_b64_list.append(b64_str)
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
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

        # Send Request
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(use_proxy))
        submit_timeout = build_submit_timeout(wait_time)
        
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
                return extract_image_from_json(openai_json, session, proxies, api_key, api_origin, timeout_val=60)
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
            except: pass
        
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
             extracted_img = extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60)
        
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
                             extracted_img = extract_image_from_json(poll_json, session, proxies, api_key, api_origin, timeout_val=60)
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
            
            extracted_img.save(out_path, **save_params)
            
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
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型选择": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k"}),
                "图片比例": (["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"], {"default": "原图1比例"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "Batch拆分模式": ("BOOLEAN", {"default": True}),
                "Batch对齐方式": (["循环补全(Max)", "裁切对齐(Min)"], {"default": "循环补全(Max)"}),
                "保存路径": ("STRING", {"default": "Shaobkj_Concurrent", "multiline": False}),
                "保存格式": (["JPEG (默认95%)", "PNG (无损)", "WEBP (无损)"], {"default": "JPEG (默认95%)"}),
                "最大并发数": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "后台最大同时执行任务数"}),
                "并发间隔": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "批量任务提交之间的间隔时间(秒)"}),
            },
            "optional": {
                 "文件名来源": ("STRING", {"forceInput": True, "multiline": False, "dynamicPrompts": False}),
                 "image_1": ("IMAGE",),
                 "image_2": ("IMAGE",),
                 "image_3": ("IMAGE",),
                 "image_4": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("API响应", "状态")
    FUNCTION = "submit_task"
    CATEGORY = "🤖shaobkj-APIbox/Concurrent"
    OUTPUT_NODE = True

    def submit_task(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 保存路径, seed, **kwargs):
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


# ----------------------------------------------------------------------------
# Node C: Load Batch Images From Path
# ----------------------------------------------------------------------------
class Shaobkj_Load_Batch_Images:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False, "placeholder": "输入文件夹路径 (如 C:\\images)"}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "限制加载数量，0为不限制"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (["numerical", "alphabetical", "date"], {"default": "numerical"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "filenames")
    FUNCTION = "load_images"
    CATEGORY = "🤖shaobkj-APIbox/Utils"

    def load_images(self, directory, image_load_cap=0, start_index=0, load_always=False, sort_method="numerical"):
        folder_path = directory
        if not folder_path or not os.path.exists(folder_path):
             raise ValueError(f"❌ 错误：文件夹路径不存在: {folder_path}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        file_list = []
        
        # 1. Scan Directory
        try:
            for f in os.listdir(folder_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(folder_path, f)
                    if os.path.isfile(full_path):
                        file_list.append(full_path)
        except Exception as e:
            raise ValueError(f"❌ 错误：读取文件夹失败: {e}")

        # 2. Sort Logic
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
             file_list.sort() # Default
        
        # 3. Apply Index & Cap
        if start_index > 0:
            if start_index >= len(file_list):
                 file_list = [] # Out of bounds
            else:
                 file_list = file_list[start_index:]
        
        if image_load_cap > 0:
            file_list = file_list[:image_load_cap]

        if not file_list:
             raise ValueError(f"❌ 错误：文件夹为空或筛选后无有效图片: {folder_path}")

        print(f"[Shaobkj-Loader] Found {len(file_list)} images in {folder_path}")

        # 4. Load Images (List Mode = Original Size)
        
        images_out = []
        masks_out = []
        filenames_out = []
        
        for file_path in file_list:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                
                # No resize - keep original size
                
                # Process Image
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                image = img.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,] # [1, H, W, C]
                
                # Process Mask
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

# ----------------------------------------------------------------------------
# Node Registration
# ----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Shaobkj_ConcurrentImageEdit_Sender": Shaobkj_ConcurrentImageEdit_Sender,
    "Shaobkj_Load_Batch_Images": Shaobkj_Load_Batch_Images
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_ConcurrentImageEdit_Sender": "🤖并发-文本-图像生成 (Sender)",
    "Shaobkj_Load_Batch_Images": "🤖批量加载图片 (Path)"
}
