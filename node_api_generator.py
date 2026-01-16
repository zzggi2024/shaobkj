import os
import json
import requests
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
from urllib.parse import urlparse

from .shaobkj_shared import get_config_value
from comfy.utils import ProgressBar


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
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k"}),
                "图片比例": (["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "API响应")
    FUNCTION = "generate_image"
    CATEGORY = "🤖shaobkj-APIbox"

    def get_resolution(self, aspect_ratio):
        ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "21:9": (1536, 640),
            "9:21": (640, 1536),
            "Free": (1024, 1024),
        }
        return ratios.get(aspect_ratio, (1024, 1024))

    def tensor2pil(self, image):
        t = image
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            t = t[0]
        if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
        return Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))

    def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def resize_pil_long_side(self, image, long_side):
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

    def generate_image(self, API密钥, API地址, 使用系统代理, 分辨率, 提示词, 图片比例, 长边设置, 等待时间, seed, **kwargs):
        api_key = API密钥
        base_url = str(API地址).rstrip("/")
        api_origin = urlparse(base_url).netloc
        resolution = 分辨率
        prompt = 提示词
        aspect_ratio = 图片比例
        long_side = 长边设置
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        seed_value = seed

        temperature = 0.7

        if not api_key:
            raise ValueError("API Key is required.")

        model_map = {
            "1k": "gemini-3-pro-image-preview",
            "2k": "gemini-3-pro-image-preview-2k",
            "4k": "gemini-3-pro-image-preview-4k",
        }
        model = model_map.get(resolution, "gemini-3-pro-image-preview")

        if not base_url.endswith("/v1"):
            base_url += "/v1"

        if aspect_ratio and aspect_ratio != "Free":
            prompt = f"{prompt} --ar {aspect_ratio}"

        input_images = []
        for i in range(1, 50):
            img_key_new = f"image_{i}"
            if img_key_new in kwargs and kwargs[img_key_new] is not None:
                input_images.append(kwargs[img_key_new])
            img_key_old = f"参考图{i}"
            if img_key_old in kwargs and kwargs[img_key_old] is not None:
                input_images.append(kwargs[img_key_old])

        width, height = self.get_resolution(aspect_ratio)

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        url = f"{base_url}/chat/completions"

        messages = []
        content = [{"type": "text", "text": prompt}]

        for img_tensor in input_images:
            if isinstance(img_tensor, torch.Tensor) and img_tensor.dim() == 4:
                batch = img_tensor.shape[0]
                for bi in range(batch):
                    pil_img = self.tensor2pil(img_tensor[bi])
                    pil_img = self.resize_pil_long_side(pil_img, long_side)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})
            else:
                pil_img = self.tensor2pil(img_tensor)
                pil_img = self.resize_pil_long_side(pil_img, long_side)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})

        messages.append({"role": "user", "content": content})

        is_dalle_image_api = "dall-e" in model.lower() and not input_images
        if is_dalle_image_api:
            url = f"{base_url.replace('/chat/completions', '')}/images/generations"
            payload = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": f"{width}x{height}",
                "response_format": "b64_json",
                "user": "comfyui-shaobkj-user",
                "seed": seed_value,
            }
        else:
            payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
            if "gemini" in model.lower():
                if aspect_ratio and aspect_ratio != "Free":
                    payload["generationConfig"] = {
                        "responseModalities": ["TEXT", "IMAGE"],
                        "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": resolution.upper()},
                        "seed": seed_value,
                    }
                else:
                    payload["generationConfig"] = {"responseModalities": ["TEXT", "IMAGE"], "seed": seed_value}

        print(f"[ComfyUI-shaobkj] Sending request to {url} with model {model}...")
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        def return_result(img_tensor, raw_text, pil_image=None):
            ui_info = {"images": []}
            if pil_image is not None:
                try:
                    filename = f"shaobkj_api_{random.randint(100000, 999999)}.png"
                    temp_dir = folder_paths.get_temp_directory()
                    full_path = os.path.join(temp_dir, filename)
                    pil_image.save(full_path)
                    ui_info["images"].append({"filename": filename, "type": "temp", "subfolder": ""})
                except Exception as e:
                    print(f"[ComfyUI-shaobkj] Error saving temp image: {e}")
            pbar.update_absolute(100)
            return {"ui": ui_info, "result": (img_tensor, raw_text)}

        session = requests.Session()
        session.trust_env = bool(使用系统代理)
        if not 使用系统代理:
            session.proxies = {}
        proxies = {} if not 使用系统代理 else None

        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout_val, verify=False, proxies=proxies)
            pbar.update_absolute(50)

            if response.status_code != 200:
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

            if "data" in res_json and isinstance(res_json["data"], list):
                data_item = res_json["data"][0]
                if "b64_json" in data_item:
                    image_data = base64.b64decode(data_item["b64_json"])
                    image = Image.open(io.BytesIO(image_data))
                    data_item["b64_json"] = "[Base64 Image Data Truncated]"
                elif "url" in data_item:
                    image_url = data_item["url"]
                    print(f"[ComfyUI-shaobkj] Downloading image from {image_url}")
                    download_timeout = 60 if timeout_val is None else timeout_val
                    auth_headers = {"Authorization": f"Bearer {api_key}"}
                    img_headers = auth_headers if urlparse(str(image_url)).netloc == api_origin else None
                    img_res = session.get(image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                    img_res.raise_for_status()
                    image = Image.open(io.BytesIO(img_res.content))
                else:
                    raise RuntimeError(f"Unknown data format in response: {json.dumps(res_json, indent=2, ensure_ascii=False)}")

                if image.mode != "RGB":
                    image = image.convert("RGB")
                return return_result(self.pil2tensor(image), json.dumps(res_json, indent=2), pil_image=image)

            if "choices" in res_json and len(res_json["choices"]) > 0:
                content_text = res_json["choices"][0]["message"].get("content", "")
                if content_text is None:
                    content_text = ""

                urls = re.findall(r"!\[.*?\]\((.*?)\)", content_text)
                if not urls:
                    urls = re.findall(r"(https?://[^\s)]+)", content_text)

                valid_image_url = None
                for u in urls:
                    if u.lower().startswith("data:"):
                        continue
                    valid_image_url = u
                    break

                if valid_image_url:
                    log_url = valid_image_url[:100] + "..." if len(valid_image_url) > 100 else valid_image_url
                    print(f"[ComfyUI-shaobkj] Found image URL in chat response: {log_url}")
                    try:
                        download_timeout = 60 if timeout_val is None else timeout_val
                        auth_headers = {"Authorization": f"Bearer {api_key}"}
                        img_headers = auth_headers if urlparse(str(valid_image_url)).netloc == api_origin else None
                        img_res = session.get(valid_image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                        img_res.raise_for_status()
                        image = Image.open(io.BytesIO(img_res.content))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        display_text = content_text[:2000] + "..." if len(content_text) > 2000 else content_text
                        return return_result(self.pil2tensor(image), display_text, pil_image=image)
                    except Exception as download_err:
                        print(f"[ComfyUI-shaobkj] Failed to download image from URL: {download_err}")

                try:
                    b64_pattern = r"data:image/[^;]+;base64,([a-zA-Z0-9+/=]+)"
                    match = re.search(b64_pattern, content_text)

                    b64_clean = ""
                    if match:
                        b64_clean = match.group(1)
                    else:
                        temp_clean = re.sub(r"^!\[.*?\]\(", "", content_text.strip())
                        temp_clean = re.sub(r"\)$", "", temp_clean)
                        temp_clean = re.sub(r"^data:image/.+;base64,", "", temp_clean)
                        b64_clean = re.sub(r"\s+", "", temp_clean)

                    if len(b64_clean) > 100:
                        try:
                            image_data = base64.b64decode(b64_clean)
                            image = Image.open(io.BytesIO(image_data))
                            if image.mode != "RGB":
                                image = image.convert("RGB")
                            if "choices" in res_json and len(res_json["choices"]) > 0:
                                res_json["choices"][0]["message"]["content"] = "[Base64 Image Data Truncated]"
                            return return_result(self.pil2tensor(image), "[Base64 Image Data Truncated]", pil_image=image)
                        except Exception as decode_err:
                            print(f"[ComfyUI-shaobkj] Base64 decode/image open failed: {decode_err}")
                except Exception:
                    pass

                print("[ComfyUI-shaobkj] No image URL or valid Base64 found in response.")
                print(f"[ComfyUI-shaobkj] Full Response: {json.dumps(res_json, indent=2, ensure_ascii=False)}")
                # Fallback: Raise error instead of returning text as image
                raise RuntimeError(f"No image found in API response. Response: {json.dumps(res_json, indent=2, ensure_ascii=False)}")

            return return_result(torch.zeros((1, 512, 512, 3)), json.dumps(res_json, indent=2, ensure_ascii=False))
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[ComfyUI-shaobkj] {error_msg}")
            raise RuntimeError(f"Generation Failed: {str(e)}") from e
