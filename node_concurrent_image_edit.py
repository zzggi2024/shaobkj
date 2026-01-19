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
    auth_headers_for_same_origin
)

# ----------------------------------------------------------------------------
# Helper Functions (Adapted from node_api_generator.py to avoid dependency)
# ----------------------------------------------------------------------------

def resize_and_encode_pil(img, long_side):
    """
    Resize PIL image and encode to base64.
    """
    if img is None:
        return None, 1.0
    
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

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
    img_resized.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str, aspect_ratio

def sanitize_text(s, max_len=1200):
    t = "" if s is None else str(s)
    t = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "data:image/...;base64,[省略]", t)
    t = re.sub(r"[A-Za-z0-9+/=]{200,}", "[省略]", t)
    if len(t) > max_len:
        t = t[:max_len] + "...(省略)"
    return t

# ----------------------------------------------------------------------------
# Background Worker
# ----------------------------------------------------------------------------

def run_concurrent_task(data):
    """
    Background task to execute the API call.
    """
    task_id_local = f"task_{int(time.time())}_{random.randint(1000,9999)}"
    print(f"[ComfyUI-shaobkj] Starting concurrent task {task_id_local}...")
    
    try:
        # 1. Parse Data
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
        
        # Collect all image paths
        image_paths = []
        if data.get("image_path"):
            image_paths.append(data.get("image_path"))
        
        # Handle additional images
        additional_images = data.get("additional_images", [])
        for item in additional_images:
            if isinstance(item, dict) and item.get("value"):
                 # Resolve path
                 try:
                     p = folder_paths.get_annotated_filepath(item.get("value"))
                     if p and os.path.exists(p):
                         image_paths.append(p)
                 except Exception:
                     pass

        if not api_key:
            raise ValueError("API Key is required")
        
        if not image_paths:
            raise ValueError(f"No valid images found.")

        # 3. Prepare API Request
        base_origin = str(api_url_base).rstrip("/")
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
        parts = [{"text": prompt}]
        
        # 2. Load and Process Images
        for idx, img_p in enumerate(image_paths):
            try:
                img = Image.open(img_p)
                img = ImageOps.exif_transpose(img) # Fix rotation
                b64_str, img_ratio = resize_and_encode_pil(img, long_side)
                if b64_str:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_str
                        }
                    })
            except Exception as e:
                print(f"[ComfyUI-shaobkj] Error processing image {img_p}: {e}")

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

        # Aspect Ratio Logic (Similar to APINode)
        if aspect_ratio != "原图1比例" and aspect_ratio != "Free":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(aspect_ratio)

        # 4. Send Request
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

        # 5. Extract Result (simplified logic for concurrent edit)
        # We try to find the image data directly or download URL
        final_image_data = None
        
        # Helper to extract
        def extract_img(json_data):
            if "candidates" in json_data:
                for cand in json_data["candidates"]:
                    for part in cand.get("content", {}).get("parts", []):
                        if "inlineData" in part:
                            return base64.b64decode(part["inlineData"]["data"])
                        if "inline_data" in part:
                            return base64.b64decode(part["inline_data"]["data"])
            return None

        final_image_data = extract_img(res_json)
        
        # If not found, check task_id for polling (simplified for this edit node)
        # NOTE: For "Edit" node, usually it returns quickly. If it enters polling, 
        # we might need full polling logic. 
        # For simplicity and "Concurrent" nature, we will try to support polling too.
        
        if not final_image_data:
            # Check for task_id
             task_id = res_json.get("id") or res_json.get("task_id")
             if not task_id and "data" in res_json:
                 task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
             if task_id:
                 print(f"[ComfyUI-shaobkj] {task_id_local}: Polling task {task_id}...")
                 poll_url = f"{url}/{task_id}"
                 poll_timeout_val = 86400 if wait_time == 0 else wait_time
                 start_poll = time.time()
                 
                 while True:
                     if (time.time() - start_poll) > poll_timeout_val:
                         raise RuntimeError("Timeout polling")
                     
                     time.sleep(2)
                     try:
                         poll_resp = session.get(poll_url, headers=headers, params={"_t": int(time.time()*1000)}, verify=False, proxies=proxies, timeout=30)
                         if poll_resp.status_code == 200:
                             poll_json = poll_resp.json()
                             final_image_data = extract_img(poll_json)
                             if final_image_data:
                                 break
                             
                             status = poll_json.get("status") or poll_json.get("task_status")
                             if status in ["FAILED", "ERROR"]:
                                 raise RuntimeError(f"Task failed: {status}")
                     except Exception as e:
                         print(f"[ComfyUI-shaobkj] Polling error: {e}")

        # 6. Save Result
        if final_image_data:
            output_dir = folder_paths.get_output_directory()
            filename = f"concurrent_edit_{int(time.time())}_{random.randint(1000,9999)}.jpg"
            out_path = os.path.join(output_dir, filename)
            
            with open(out_path, "wb") as f:
                f.write(final_image_data)
            
            print(f"[ComfyUI-shaobkj] {task_id_local}: Success! Saved to {out_path}")
            # Send socket event to notify UI? (Optional, but nice)
            PromptServer.instance.send_sync("shaobkj.concurrent.success", {"task_id": task_id_local, "filename": filename})
        else:
            raise RuntimeError("No image data found in response")

    except Exception as e:
        err_msg = f"Error: {str(e)}"
        print(f"[ComfyUI-shaobkj] {task_id_local}: {err_msg}")
        traceback.print_exc()
        PromptServer.instance.send_sync("shaobkj.concurrent.error", {"task_id": task_id_local, "error": err_msg})


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
                "image": ("STRING", {"image_upload": True}),  # Upload widget
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

    def execute_concurrent_task(self, image, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 图片比例, 保存路径, **kwargs):
        """
        Main execution function called by ComfyUI Queue.
        This function captures inputs (including connected tensors), prepares data, 
        and starts the background thread.
        """
        
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
            "image_name": image, # Uploaded image name
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
        # 1. From Upload (path)
        # 2. From Tensor (PIL objects)
        
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
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
        parts = [{"text": prompt}]
        
        for img in pil_images:
            try:
                b64_str, img_ratio = resize_and_encode_pil(img, long_side)
                if b64_str:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_str
                        }
                    })
            except Exception as e:
                print(f"[ComfyUI-shaobkj] Error encoding image: {e}")

        # ... (Rest of the logic: Seed, Payload, Request, Polling, Save) ...
        # Copied from previous run_concurrent_task but adapted variables
        
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

        # Extract Result
        def extract_img(json_data):
            if "candidates" in json_data:
                for cand in json_data["candidates"]:
                    for part in cand.get("content", {}).get("parts", []):
                        if "inlineData" in part:
                            return base64.b64decode(part["inlineData"]["data"])
                        if "inline_data" in part:
                            return base64.b64decode(part["inline_data"]["data"])
            return None

        final_image_data = extract_img(res_json)
        
        if not final_image_data:
             task_id = res_json.get("id") or res_json.get("task_id")
             if not task_id and "data" in res_json:
                 task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
             
             if task_id:
                 print(f"[ComfyUI-shaobkj] {task_id_local}: Polling task {task_id}...")
                 poll_url = f"{url}/{task_id}"
                 poll_timeout_val = 86400 if wait_time == 0 else wait_time
                 start_poll = time.time()
                 
                 while True:
                     if (time.time() - start_poll) > poll_timeout_val:
                         raise RuntimeError("Timeout polling")
                     
                     time.sleep(2)
                     try:
                         poll_resp = session.get(poll_url, headers=headers, params={"_t": int(time.time()*1000)}, verify=False, proxies=proxies, timeout=30)
                         if poll_resp.status_code == 200:
                             poll_json = poll_resp.json()
                             final_image_data = extract_img(poll_json)
                             if final_image_data:
                                 break
                             
                             status = poll_json.get("status") or poll_json.get("task_status")
                             if status in ["FAILED", "ERROR"]:
                                 raise RuntimeError(f"Task failed: {status}")
                     except Exception as e:
                         print(f"[ComfyUI-shaobkj] Polling error: {e}")

        # Save Result
        if final_image_data:
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
            
            with open(out_path, "wb") as f:
                f.write(final_image_data)
            
            print(f"[ComfyUI-shaobkj] {task_id_local}: Success! Saved to {out_path}")
            PromptServer.instance.send_sync("shaobkj.concurrent.success", {"task_id": task_id_local, "filename": filename, "path": out_path})
        else:
            raise RuntimeError("No image data found in response")

    except Exception as e:
        err_msg = f"Error: {str(e)}"
        print(f"[ComfyUI-shaobkj] {task_id_local}: {err_msg}")
        traceback.print_exc()
        PromptServer.instance.send_sync("shaobkj.concurrent.error", {"task_id": task_id_local, "error": err_msg})


# Maintain backward compatibility for the pure API route if needed, 
# but now we primarily use the Queue execution. 
# We can wrap run_concurrent_task_internal for the API route too.
def run_concurrent_task(data):
    run_concurrent_task_internal(data)

