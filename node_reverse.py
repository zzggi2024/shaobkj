import json
import requests
import numpy as np
from PIL import Image
import io
import base64
import random
import time
import traceback

from .shaobkj_shared import get_config_value
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "谷歌搜索": ("BOOLEAN", {"default": False}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("提示词", "API响应")
    FUNCTION = "inference"
    CATEGORY = "🤖shaobkj-APIbox"

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

    def inference(self, API密钥, API地址, 模型名称, 系统提示词, 需求提示词, 使用系统代理, 长边设置, 等待时间, seed, 谷歌搜索, **kwargs):
        api_key = API密钥
        base_url = str(API地址).rstrip("/")
        model = 模型名称
        system_prompt = 系统提示词.strip() if isinstance(系统提示词, str) else ""
        user_prompt = 需求提示词.strip() if isinstance(需求提示词, str) else ""
        seed_value = seed
        prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt + f"\n\n[Seed]\n{seed_value}"
        timeout_val = None if int(等待时间) == 0 else int(等待时间)

        def extract_error(obj):
            code = None
            message = None
            cur = obj
            for _ in range(3):
                if isinstance(cur, dict):
                    code = cur.get("code") or code
                    message = cur.get("message") if cur.get("message") is not None else message
                    if isinstance(message, str):
                        s = message.strip()
                        if s.startswith("{") and s.endswith("}"):
                            try:
                                cur = json.loads(s)
                                continue
                            except Exception:
                                pass
                    if isinstance(message, dict):
                        cur = message
                        continue
                break
            return code, message

        def raise_if_quota_error(status_code, payload):
            code, message = extract_error(payload)
            if code == "quota_not_enough":
                raise RuntimeError("API 额度不足（quota_not_enough），请充值或更换 API Key。")
            if code == "fail_to_fetch_task":
                inner_code, inner_message = extract_error(message)
                if inner_code == "quota_not_enough":
                    raise RuntimeError("API 额度不足（quota_not_enough），请充值或更换 API Key。")
                if isinstance(inner_message, str) and "quota_not_enough" in inner_message:
                    raise RuntimeError("API 额度不足（quota_not_enough），请充值或更换 API Key。")
            if isinstance(message, str) and "quota_not_enough" in message:
                raise RuntimeError("API 额度不足（quota_not_enough），请充值或更换 API Key。")
            
            # Gemini specific error handling
            if isinstance(payload, dict) and "error" in payload:
                err = payload["error"]
                if isinstance(err, dict):
                     msg = err.get("message", "")
                     if "quota" in msg.lower() or "limit" in msg.lower():
                          print(f"[ComfyUI-shaobkj] Possible quota error: {msg}")

            raise RuntimeError(f"API Error {status_code}: {payload}")

        input_images = []
        for i in range(1, 50):
            img_key = f"image_{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                input_images.append(kwargs[img_key])
        if "图像" in kwargs and kwargs["图像"] is not None:
            input_images.append(kwargs["图像"])

        if not api_key:
            raise ValueError("API Key is required.")

        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"

        parts = [{"text": prompt}]

        if len(input_images) > 0:
            for img_tensor_batch in input_images:
                batch_size = img_tensor_batch.shape[0]
                for i in range(batch_size):
                    img_tensor = img_tensor_batch[i]
                    img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    pil_img = self.resize_pil_long_side(pil_img, 长边设置)

                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="JPEG", quality=90)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

        payload = {"contents": [{"role": "user", "parts": parts}]}
        if 谷歌搜索:
            payload["tools"] = [{"googleSearch": {}}]

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        print(f"[ComfyUI-shaobkj] Sending inference request to {base_url} (Model: {model})...")
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        session = requests.Session()
        session.trust_env = bool(使用系统代理)
        proxies = {} if not 使用系统代理 else None

        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout_val, verify=False, proxies=proxies)
            if response.status_code != 200:
                print(f"[ComfyUI-shaobkj] API Error: {response.status_code}")
                try:
                    err_msg = response.json()
                except Exception:
                    err_msg = response.text
                raise_if_quota_error(response.status_code, err_msg)

            res_json = response.json()
            pbar.update_absolute(60)

            generated_text = ""
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                candidate = res_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]

            if not generated_text:
                generated_text = "No text response generated."
            pbar.update_absolute(100)
            return (generated_text, json.dumps(res_json, ensure_ascii=False))
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[ComfyUI-shaobkj] Inference Error: {error_msg}")
            raise RuntimeError(f"Inference Failed: {str(e)}") from e
