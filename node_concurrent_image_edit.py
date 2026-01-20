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
    sanitize_text
)

# ----------------------------------------------------------------------------
# API Route
# ----------------------------------------------------------------------------

@PromptServer.instance.routes.post("/shaobkj/concurrent/submit")
async def api_concurrent_submit(request):
    try:
        json_data = await request.json()
        
        # Resolve image path
        image_name = json_data.get("image_name")
        if image_name:
            # ComfyUI image upload widget returns just filename usually, stored in input dir
            image_path = folder_paths.get_annotated_filepath(image_name)
            json_data["image_path"] = image_path
        
        # Start background thread
        t = threading.Thread(target=run_concurrent_task, args=(json_data,))
        t.daemon = True
        t.start()
        
        return web.json_response({"status": "success", "message": "Task started in background"})
        
    except Exception as e:
        return web.json_response({"status": "error", "message": str(e)}, status=500)


# ----------------------------------------------------------------------------
# Node Definition
# ----------------------------------------------------------------------------

class Shaobkj_ConcurrentImageEdit:
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
                    ["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21", "原图1比例"],
                    {"default": "原图1比例"},
                ),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "保存路径": ("STRING", {"default": "", "multiline": False, "placeholder": "默认为 output 目录 (Default output dir)"}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                 "image_1": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("API响应",)
    FUNCTION = "execute_concurrent_task"
    CATEGORY = "🤖shaobkj-APIbox"
    OUTPUT_NODE = True

    def execute_concurrent_task(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 保存路径, **kwargs):
        """
        Main execution function called by ComfyUI Queue.
        This function captures inputs (including connected tensors), prepares data, 
        and starts the background thread.
        """
        
        # 0. Pre-check (Main Thread) to prevent multiple popups
        if not API密钥 or str(API密钥).strip() == "":
            raise ValueError("❌ 错误：API Key 不能为空 (API Key is required)")
        
        # Check for non-ascii characters in API Key
        try:
            str(API密钥).encode('ascii')
        except UnicodeEncodeError:
            raise ValueError("❌ 错误：API Key 包含非法字符（如中文）。请检查是否复制了多余内容。")

        if not API地址 or str(API地址).strip() == "":
            raise ValueError("❌ 错误：API 地址不能为空 (API URL is required)")

        # Check for non-ascii in API URL
        try:
            str(API地址).encode('ascii')
        except UnicodeEncodeError:
             raise ValueError("❌ 错误：API 地址包含非法字符（如中文）。")
             
        if not str(API地址).startswith("http"):
             raise ValueError("❌ 错误：API 地址必须以 http:// 或 https:// 开头")
        
        # 1. Prepare Data
        data = {
            "api_key": API密钥,
            "api_url": API地址,
            "model": 模型选择,
            "use_proxy": 使用系统代理,
            "resolution": 分辨率,
            "prompt": 提示词,
            "aspect_ratio": 图片比例,
            "long_side": int(kwargs.get("输入图像-长边设置", 1280)),
            "wait_time": int(kwargs.get("等待时间", 180)),
            "seed": int(kwargs.get("seed", 0)),
            "image_name": None, # Removed upload widget
            "tensor_images": [],
            "save_path": 保存路径
        }

        # 2. Collect Connected Images (Tensor)
        # We need to handle image_1, image_2, etc. from kwargs
        for k, v in kwargs.items():
            if k.startswith("image_") and isinstance(v, torch.Tensor):
                # Store tensor directly to be processed in background (or pre-process here)
                # Tensors can be passed to threads, but better to convert to PIL/Buffer here to avoid thread safety issues with CUDA tensors?
                # Actually, CUDA tensors in other threads might be tricky. 
                # Let's convert to PIL here in main thread.
                
                # Handle batch dimension [B, H, W, C]
                for i in range(v.shape[0]):
                     # Convert single image tensor to PIL
                     t = v[i]
                     # Ensure [H, W, C]
                     if t.dim() == 3:
                         pil_img = Image.fromarray(np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8))
                         data["tensor_images"].append(pil_img)

        # 3. Start Background Thread
        # We pass the data dict which now contains PIL images (safe for threading)
        t = threading.Thread(target=run_concurrent_task_internal, args=(data,))
        t.daemon = True
        t.start()

        # Return status message only
        status_msg = f"任务已后台启动。\n保存路径: {保存路径 if 保存路径 else 'Output Dir'}\n(结果将异步保存，无法在此预览)"
        
        return (status_msg,)

# Refactored worker to handle both path and PIL objects
def run_concurrent_task_internal(data):
    task_id_local = f"task_{int(time.time())}_{random.randint(1000,9999)}"
    print(f"[ComfyUI-shaobkj] Starting concurrent task {task_id_local}...")
    
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

        if aspect_ratio != "原图1比例" and aspect_ratio != "Free":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(aspect_ratio)

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
        
        task_id = None
        
        if not extracted_img:
             task_id = res_json.get("id") or res_json.get("task_id")
             if not task_id and "data" in res_json:
                 task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
             if task_id:
                 print(f"[ComfyUI-shaobkj] {task_id_local}: Polling task {task_id}...")
                 poll_url = f"{url}/{task_id}"
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
                                 raise RuntimeError(f"Task failed: {status}")
                     except Exception as e:
                         fail_count += 1
                         print(f"[ComfyUI-shaobkj] Polling error: {e}")
                         if fail_count > 10:
                             raise

        # Save Result
        if extracted_img:
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
            
            # Record success
            save_local_record("Concurrent_Edit", str(task_id or task_id_local), "Success", api_url_base)
            
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
        traceback.print_exc()
        PromptServer.instance.send_sync("shaobkj.concurrent.error", {"task_id": task_id_local, "error": err_msg})


# Maintain backward compatibility for the pure API route if needed, 
# but now we primarily use the Queue execution. 
# We can wrap run_concurrent_task_internal for the API route too.
def run_concurrent_task(data):
    run_concurrent_task_internal(data)

