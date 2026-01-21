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
    
    print(f"[ComfyUI-shaobkj] Starting concurrent task {task_id_local}...")
    
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
        wait_time = int(data.get("wait_time", 180))
        seed_val = int(data.get("seed", 0))
        save_path_input = data.get("save_path", "")

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
                 print(f"[ComfyUI-shaobkj] Error loading uploaded image: {e}")

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
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
        # Force generation instruction
        final_prompt = str(prompt) + "\n\n(Generate an image based on this description)"
        parts = [{"text": final_prompt}]
        
        for img in pil_images:
            try:
                # Use shared helper
                b64_str, img_ratio = resize_and_encode_image(img, long_side)
                if b64_str:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_str
                        }
                    })
            except Exception as e:
                print(f"[ComfyUI-shaobkj] Error encoding image: {e}")

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
        response.raise_for_status()
        res_json = response.json()

        # Extract Result using shared helper
        extracted_img = extract_image_from_json(res_json, session, proxies, api_key, api_origin, timeout_val=60)
        
        remote_task_id = None
        
        if not extracted_img:
             remote_task_id = res_json.get("id") or res_json.get("task_id")
             if not remote_task_id and "data" in res_json:
                 remote_task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
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
                         print(f"[ComfyUI-shaobkj] Polling error: {e}")
                         if fail_count > 10:
                             raise

        # Save Result
        if extracted_img:
            # Determine Filename
            custom_filename = data.get("output_filename")
            if custom_filename:
                # Clean up path to get just basename without extension
                base_name = os.path.splitext(os.path.basename(str(custom_filename)))[0]
                # Keep extension as jpg since we save as JPEG
                filename = f"{base_name}.jpg"
            else:
                filename = f"concurrent_edit_{int(time.time())}_{random.randint(1000,9999)}.jpg"
            
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
                    print(f"[ComfyUI-shaobkj] Failed to create custom dir {custom_dir}, using default. Error: {e}")

            out_path = os.path.join(out_dir, filename)
            
            extracted_img.save(out_path, format="JPEG", quality=95)
            
            print(f"[ComfyUI-shaobkj] {task_id_local}: Success! Saved to {out_path}")
            
            # Record success in async manager
            update_async_task(task_id_local, {
                "status": "success",
                "image_path": out_path,
                "completed_at": int(time.time())
            })
            
            # Also record in local log file
            save_local_record("Concurrent_Edit", str(remote_task_id or task_id_local), "Success", api_url_base)
            
            PromptServer.instance.send_sync("shaobkj.concurrent.success", {"task_id": task_id_local, "filename": filename, "path": out_path})
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
             
        print(f"[ComfyUI-shaobkj] {task_id_local}: {err_msg}")
        
        # Record failure in async manager
        update_async_task(task_id_local, {
            "status": "failed",
            "error": err_msg,
            "completed_at": int(time.time())
        })

        traceback.print_exc()
        PromptServer.instance.send_sync("shaobkj.concurrent.error", {"task_id": task_id_local, "error": err_msg})


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
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "Batch拆分模式": ("BOOLEAN", {"default": False}),
                "Batch对齐方式": (["循环补全(Max)", "裁切对齐(Min)"], {"default": "循环补全(Max)"}),
                "保存路径": ("STRING", {"default": "", "multiline": False, "placeholder": "默认为 output 目录"}),
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
    RETURN_NAMES = ("任务ID", "状态")
    FUNCTION = "submit_task"
    CATEGORY = "🤖shaobkj-APIbox/Concurrent"
    OUTPUT_NODE = True

    def submit_task(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 保存路径, **kwargs):
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
            # Handle images: they are List of Tensors (if Batch) or List of Lists (if List input)?
            # If INPUT_IS_LIST=True:
            # - If upstream is Batch Tensor [B,H,W,C], we get [Tensor(B,H,W,C)] (List of length 1)
            # - If upstream is List [Tensor(1,H,W,C), Tensor(1,H,W,C)], we get [Tensor, Tensor] (List of length N)
            # This unifies both worlds!
            clean_kwargs[k] = v

        long_side_val = int(get_val(kwargs.get("输入图像-长边设置", [1280])))
        wait_time_val = int(get_val(kwargs.get("等待时间", [180])))
        seed_val = int(get_val(kwargs.get("seed", [0])))
        batch_split_val = get_val(kwargs.get("Batch拆分模式", [False]))
        batch_align_val = get_val(kwargs.get("Batch对齐方式", ["循环补全(Max)"]))
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
            "save_path": save_path_val
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
        # With INPUT_IS_LIST=True, "Legacy Mode" (Single Request) means:
        # Take ALL images from ALL inputs and put them into ONE request.
        
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
                    # Normalize whatever we got into flat list of tensors
                    tensors = normalize_image_input(v)
                    for t in tensors:
                        if t.dim() == 3:
                            pil_img = Image.fromarray(np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8))
                            data["tensor_images"].append(pil_img)
            
            # Handle Filename
            # Flatten filename list if present
            f_list = []
            if filename_source_val:
                if isinstance(filename_source_val, list):
                    for f in filename_source_val:
                        if isinstance(f, str) and "\n" in f:
                            f_list.extend(f.split("\n"))
                        else:
                            f_list.append(str(f))
                elif isinstance(filename_source_val, str):
                    f_list.append(filename_source_val)
            
            if f_list:
                data["output_filename"] = f_list[0]

            # Start Thread
            t = threading.Thread(target=run_concurrent_task_internal, args=(data,))
            t.daemon = True
            t.start()
            
            # Return list of length 1 (because INPUT_IS_LIST=True expects list output)
            return ([base_task_id], [f"已提交: {base_task_id}"])

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
            
            # Debugging Info (Console)
            debug_msg = "[Shaobkj-Debug] Batch Inputs: "
            for k, v in normalized_inputs.items():
                debug_msg += f"{k}={len(v)}, "
            
            # Handle prompt list
            # Prompt input is a list of strings (if multiline/batch)
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
            
            # Determine Final Batch Size
            
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
            
            # Create a summary string
            ui_summary = f"BatchTotal={final_batch_size} [{batch_align_val}] ["
            details = []
            for k, v in normalized_inputs.items():
                details.append(f"{k}:{len(v)}")
            if len(prompts) > 1:
                details.append(f"Prompt:{len(prompts)}")
            if filename_list:
                details.append(f"Filename:{len(filename_list)}")
            ui_summary += ", ".join(details) + "]"

            generated_ids = []
            
            for i in range(final_batch_size):
                # Generate unique ID for this sub-task
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
                        # Modulo index
                        idx = i % len(tensor_list)
                        t = tensor_list[idx]
                        if t.dim() == 3:
                            pil_img = Image.fromarray(np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8))
                            sub_data["tensor_images"].append(pil_img)
                
                # Start Thread
                t = threading.Thread(target=run_concurrent_task_internal, args=(sub_data,))
                t.daemon = True
                t.start()
                
                generated_ids.append(sub_task_id)
            
            # Return list of IDs (Since INPUT_IS_LIST=True, output should be list)
            status_list = [f"已提交: {gid} | {ui_summary}" for gid in generated_ids]
            
            # CRITICAL FIX for ComfyUI "List Execution":
            # If we return a LIST of strings, ComfyUI might interpret this as "Run downstream node N times".
            # BUT, we want downstream node (Receiver) to run N times ANYWAY because we are outputting N IDs.
            # So returning a list IS correct for "Batch" behavior.
            
            # HOWEVER, if `generated_ids` is empty (e.g. no inputs), we should return empty list or handle gracefully.
            if not generated_ids:
                return ([], [])
            
            return (generated_ids, status_list)


# ----------------------------------------------------------------------------
# Node B: Receiver (Async)
# ----------------------------------------------------------------------------

class Shaobkj_ConcurrentImageEdit_Receiver:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "任务ID": ("STRING", {"multiline": False, "default": "", "placeholder": "连接发送端输出或留空自动获取"}),
                 "阻塞等待结果": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "状态")
    FUNCTION = "receive_result"
    CATEGORY = "🤖shaobkj-APIbox/Concurrent"

    def receive_result(self, 任务ID="", 阻塞等待结果=True):
        # Safeguard: Handle list input if batching doesn't unwrap
        # With INPUT_IS_LIST=True in Sender, Sender outputs a list of IDs.
        # But Receiver does NOT have INPUT_IS_LIST=True.
        # So ComfyUI should iterate Receiver for each ID in the list.
        # However, if the list is empty or something weird happens...
        
        target_id_input = 任务ID
        if isinstance(target_id_input, list):
             if len(target_id_input) > 0:
                 target_id_input = target_id_input[0]
             else:
                 target_id_input = ""
        
        # Retry loop for blocking mode
        timeout = 300 # 5 minutes max wait for blocking
        start_time = time.time()
        
        while True:
            tasks = get_all_async_tasks()
            target_task = None
            target_id = None
            is_running = False

            # 1. If ID provided, check specific task
            if target_id_input:
                if target_id_input in tasks:
                    t = tasks[target_id_input]
                    status = t.get("status")
                    if status == "success" and not t.get("downloaded"):
                        target_task = t
                        target_id = target_id_input
                    elif status == "running":
                        is_running = True
                    elif status == "success":
                         # Already downloaded?
                         # If we are in a batch run, maybe we are re-checking?
                         # But we want to return the image anyway if we have the ID.
                         # Wait, logic below says "and not t.get('downloaded')".
                         # If it's already downloaded, we skip it?
                         # If I provide a specific ID, I want that result, even if downloaded before!
                         # The "downloaded" flag is mainly for the "Auto Pick Oldest" mode.
                         target_task = t
                         target_id = target_id_input
                else:
                    # Task ID might not be written to JSON yet if Sender just spawned it?
                    # Sender writes "running" immediately.
                    pass
            
            # 2. If no ID or not ready, find oldest successful task (Only if ID not provided)
            # If ID IS provided, we only care about that ID.
            if not target_task and not target_id_input:
                # Sort by submitted_at
                sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].get("submitted_at", 0))
                for tid, t in sorted_tasks:
                    if t.get("status") == "success" and not t.get("downloaded"):
                        target_task = t
                        target_id = tid
                        break
            
            # 3. Decision
            if target_task:
                # Found result! Break loop and process.
                break
            
            if 阻塞等待结果 and target_id_input and is_running:
                # If we are blocking, have a specific ID, and it's running -> Wait.
                if time.time() - start_time > timeout:
                    return (torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"等待超时: {target_id_input}")
                time.sleep(1)
                continue
            elif 阻塞等待结果 and target_id_input and not target_task:
                 # ID provided, but task not found or failed?
                 # Maybe it's just starting up?
                 if time.time() - start_time < 5:
                     time.sleep(1)
                     continue
                 return (torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"未找到任务: {target_id_input}")
            else:
                # Not blocking, or not running (failed?), or no ID provided and no results ready.
                # Return empty.
                empty_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                status_msg = "等待中: 任务运行中" if is_running else "等待中: 无可用结果"
                return (empty_img, status_msg)

        # 4. Load Image (Target Found)
        image_path = target_task.get("image_path")
        if not image_path or not os.path.exists(image_path):
             return (torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"错误: 文件丢失 {target_id}")

        try:
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Convert to Tensor
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            
            # Mark as downloaded ONLY if we auto-picked it.
            # If user provided ID, we shouldn't mark it as "consumed" for others?
            # Or we should?
            # If we mark it downloaded, next auto-pick won't get it. That's good.
            update_async_task(target_id, {"downloaded": True})
            
            return (img_tensor, f"已获取: {target_id}")
            
        except Exception as e:
            print(f"[Shaobkj-Receiver] Error loading image: {e}")
            return (torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"加载失败: {e}")

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
             # Extract numbers and sort, ignoring common separators to unify "Name (1)" and "Name1"
             # Use (type_priority, value) tuple to prevent TypeError when comparing int vs str
             def natural_sort_key(s):
                 parts = re.split('([0-9]+)', s)
                 processed = []
                 for text in parts:
                     if text.isdigit():
                         # Priority 0 for numbers
                         processed.append((0, int(text)))
                     else:
                         # Priority 1 for strings
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
             # Just return empty or raise? Inspire usually raises if empty or returns empty batch?
             # Let's raise to be clear
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

        # Return Lists (Implicitly "Original Size")
        # CRITICAL FIX: To prevent ComfyUI from iterating this list, we must return it as a SINGLE item (a list wrapped in another structure?)
        # NO! ComfyUI iterates if the return type matches the connection type and it's a list.
        # But if the downstream node has INPUT_IS_LIST = True, ComfyUI passes the whole list.
        
        # The issue in screenshot: "Batch Inputs: image_1=0"
        # This means `image_1` received an EMPTY list or empty tensor?
        # But `Shaobkj-Loader` printed "Found 8 images".
        # So `images_out` has 8 items.
        # Why did Sender receive 0 items?
        
        # Hypothesis:
        # Sender `normalize_image_input` failed to extract tensors.
        # `images_out` is `[Tensor, Tensor...]`.
        # Sender receives this list.
        # `normalize_image_input` iterates it.
        # It checks `isinstance(item, torch.Tensor)`.
        # Yes, item is Tensor [1, H, W, C].
        # It appends `item[i]`. `item.shape[0]` is 1. So it appends `item[0]`.
        # `flat_list` should have 8 items.
        
        # Wait, look at the log: `image_1=0`.
        # This means `normalized_inputs['image_1']` is empty.
        # Why?
        # Maybe `INPUT_IS_LIST=True` causes ComfyUI to wrap the list AGAIN?
        # If Loader returns `[T1, T2]`.
        # ComfyUI passes `[[T1, T2]]` (List of List)?
        # Let's debug `normalize_image_input`.
        # I added check: `if isinstance(val, list): for item in val: ...`
        # If `val` is `[[T1, T2]]`.
        # Item 0 is `[T1, T2]`.
        # `isinstance([T1, T2], torch.Tensor)` is False.
        # So it skips!
        # Ah! `normalize_image_input` only handles List of Tensors OR Tensor.
        # It does NOT handle List of Lists!
        
        # If upstream returns a List, and INPUT_IS_LIST=True, ComfyUI might pass the list AS IS.
        # But if upstream returns a Batch Tensor, ComfyUI passes [Tensor].
        
        # Let's fix `normalize_image_input` to handle nested lists recursively or just one level deeper.
        
        return (images_out, masks_out, filenames_out)

# ----------------------------------------------------------------------------
# Legacy Node (Wrapper for compatibility)
# ----------------------------------------------------------------------------
class Shaobkj_ConcurrentImageEdit(Shaobkj_ConcurrentImageEdit_Sender):
    @classmethod
    def INPUT_TYPES(s):
        # Return same as Sender but maybe adapt category if needed
        d = Shaobkj_ConcurrentImageEdit_Sender.INPUT_TYPES()
        return d
    
    CATEGORY = "🤖shaobkj-APIbox" # Keep original category for old workflows
