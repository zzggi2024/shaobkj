import json
import numpy as np
import torch
from PIL import Image
import io
import base64
import traceback
import requests

from .shaobkj_shared import (
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    post_json_with_retry,
    resize_pil_long_side,
    get_video_file_path,
    compress_video_to_base64,
)
from comfy.utils import ProgressBar


class Shaobkj_Reverse_Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "系统提示词": ("STRING", {"multiline": True, "default": ""}),
                "需求提示词": ("STRING", {"multiline": True, "default": "Describe this content in detail to recreate it as a prompt."}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型名称": (["gemini-2.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"], {"default": "gemini-2.5-flash"}),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "谷歌搜索": ("BOOLEAN", {"default": False}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("提示词", "API响应")
    FUNCTION = "inference"
    CATEGORY = "🤖shaobkj-APIbox"

    def inference(self, API密钥, API地址, 模型名称, 系统提示词, 需求提示词, 使用系统代理, 长边设置, 等待时间, seed, 谷歌搜索, **kwargs):
        api_key = API密钥
        base_url = str(API地址).rstrip("/")
        model = 模型名称
        system_prompt = 系统提示词.strip() if isinstance(系统提示词, str) else ""
        user_prompt = 需求提示词.strip() if isinstance(需求提示词, str) else ""
        seed_value = seed
        
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        submit_timeout = build_submit_timeout(int(等待时间))
        
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))

        # ---------------------------------------------------------------------------
        # 1. Prepare Content (Images & Video)
        # ---------------------------------------------------------------------------
        image_assets = [] # List of base64 strings
        video_asset = None # Base64 string for video
        
        # Process Images
        input_images = []
        for i in range(1, 50):
            img_key = f"image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                input_images.append(kwargs[img_key])
        if "图像" in kwargs and kwargs["图像"] is not None:
            input_images.append(kwargs["图像"])
            
        if len(input_images) > 0:
            for img_tensor_batch in input_images:
                batch_size = img_tensor_batch.shape[0]
                for i in range(batch_size):
                    img_tensor = img_tensor_batch[i]
                    if isinstance(img_tensor, torch.Tensor):
                        img_u8 = (img_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    else:
                        img_u8 = np.clip(255.0 * np.array(img_tensor), 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_u8)
                    pil_img = resize_pil_long_side(pil_img, 长边设置)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG", quality=85)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_assets.append(img_b64)

        # Process Video
        if "video" in kwargs and kwargs["video"] is not None:
            video_path = get_video_file_path(kwargs["video"])
            if video_path:
                print(f"[ComfyUI-shaobkj] Processing video for inference: {video_path}")
                # Compress video (limit to 15s to be safe for prompt extraction)
                video_asset = compress_video_to_base64(video_path, max_size_mb=20, duration_limit=15)
                if video_asset:
                    print(f"[ComfyUI-shaobkj] Video encoded successfully ({len(video_asset)//1024} KB)")
                else:
                    print(f"[ComfyUI-shaobkj] Video encoding failed.")

        # ---------------------------------------------------------------------------
        # 2. Define Protocols (OpenAI First, then Gemini)
        # ---------------------------------------------------------------------------
        protocols = []
        
        # Helper: Determine OpenAI base URL
        # If url ends with /v1, remove it to get root, then append /v1/chat/completions
        # If url doesn't end with /v1, assume it's root, append /v1/chat/completions
        openai_base = base_url[:-3] if base_url.endswith("/v1") else base_url
        openai_url = f"{openai_base}/v1/chat/completions"
        
        # --- Protocol A: OpenAI Compatible ---
        openai_content = []
        if user_prompt:
            openai_content.append({"type": "text", "text": user_prompt})
        
        for img_b64 in image_assets:
            openai_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            
        if video_asset:
            # OpenAI format extension for video (commonly supported by Gemini wrappers)
            openai_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_asset}"}
            })
            
        openai_messages = []
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        openai_messages.append({"role": "user", "content": openai_content})
        
        openai_payload = {
            "model": model,
            "messages": openai_messages,
            "seed": seed_value if seed_value >= 0 else None,
            "stream": False,
            "max_tokens": 4096
        }
        # Clean payload
        openai_payload = {k: v for k, v in openai_payload.items() if v is not None}
        
        protocols.append({
            "name": "OpenAI Compatible",
            "url": openai_url,
            "headers": {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            "payload": openai_payload,
            "parser": "openai"
        })
        
        # --- Protocol B: Gemini Native ---
        gemini_url = f"{openai_base}/v1beta/models/{model}:generateContent"
        
        gemini_parts = []
        full_prompt = (system_prompt + "\n\n" + user_prompt).strip()
        if full_prompt:
            gemini_parts.append({"text": full_prompt})
            
        for img_b64 in image_assets:
            gemini_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64
                }
            })
            
        if video_asset:
             gemini_parts.append({
                "inline_data": {
                    "mime_type": "video/mp4",
                    "data": video_asset
                }
            })
            
        gemini_payload = {
            "contents": [{"role": "user", "parts": gemini_parts}],
            "generationConfig": {
                "seed": seed_value if seed_value >= 0 else 0,
                "maxOutputTokens": 4096
            }
        }
        if 谷歌搜索:
            gemini_payload["tools"] = [{"googleSearch": {}}]
            
        protocols.append({
            "name": "Gemini Native",
            "url": gemini_url,
            "headers": {"Content-Type": "application/json", "x-goog-api-key": api_key},
            "payload": gemini_payload,
            "parser": "gemini"
        })

        # ---------------------------------------------------------------------------
        # 3. Execute Protocols
        # ---------------------------------------------------------------------------
        last_error = None
        pbar = ProgressBar(100)
        pbar.update_absolute(0)
        
        for idx, proto in enumerate(protocols):
            print(f"[ComfyUI-shaobkj] Trying Protocol: {proto['name']} ({proto['url']})...")
            try:
                # Use a shorter timeout for the first attempt if we have fallback? 
                # No, give it proper time.
                response = post_json_with_retry(
                    session,
                    proto['url'],
                    headers=proto['headers'],
                    payload=proto['payload'],
                    timeout=submit_timeout,
                    proxies=proxies,
                    verify=False,
                    max_retries=1 # Retry logic is handled by switching protocols
                )
                
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                        pbar.update_absolute(100)
                        
                        # Parse Result
                        generated_text = ""
                        if proto['parser'] == "openai":
                            if "choices" in res_json and len(res_json["choices"]) > 0:
                                generated_text = res_json["choices"][0]["message"]["content"]
                        elif proto['parser'] == "gemini":
                            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                                parts = res_json["candidates"][0].get("content", {}).get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        generated_text += part["text"]
                        
                        if not generated_text:
                            # Try generic extraction if parser specific failed
                            print(f"[ComfyUI-shaobkj] Warning: Parser {proto['parser']} yielded no text. Dumping JSON.")
                            generated_text = json.dumps(res_json, ensure_ascii=False)
                            
                        return (generated_text, json.dumps(res_json, ensure_ascii=False))
                        
                    except json.JSONDecodeError:
                        print(f"[ComfyUI-shaobkj] Invalid JSON from {proto['name']}")
                else:
                    print(f"[ComfyUI-shaobkj] {proto['name']} Failed: HTTP {response.status_code}")
                    try:
                        err_body = response.text
                    except:
                        err_body = "Unknown Error"
                    last_error = f"{proto['name']} HTTP {response.status_code}: {err_body[:200]}"
                    
            except Exception as e:
                print(f"[ComfyUI-shaobkj] {proto['name']} Exception: {e}")
                last_error = f"{proto['name']} Error: {str(e)}"
                
            # If we are here, this protocol failed. Loop continues to next protocol.
        
        # All protocols failed
        pbar.update_absolute(100)
        error_msg = f"All inference protocols failed.\nLast Error: {last_error}"
        print(f"[ComfyUI-shaobkj] {error_msg}")
        raise RuntimeError(error_msg)
