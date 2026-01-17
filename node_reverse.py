import json
import numpy as np
import torch
from PIL import Image
import io
import base64
import traceback

from .shaobkj_shared import (
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    post_json_with_retry,
    resize_pil_long_side,
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

    def inference(self, API密钥, API地址, 模型名称, 系统提示词, 需求提示词, 使用系统代理, 长边设置, 等待时间, seed, 谷歌搜索, **kwargs):
        api_key = API密钥
        base_url = str(API地址).rstrip("/")
        model = 模型名称
        system_prompt = 系统提示词.strip() if isinstance(系统提示词, str) else ""
        user_prompt = 需求提示词.strip() if isinstance(需求提示词, str) else ""
        seed_value = seed
        prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
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

        url = f"{base_url}/v1beta/models/{model}:generateContent"

        parts = [{"text": prompt}]

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
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})

        payload = {"contents": [{"role": "user", "parts": parts}]}
        # 将 seed 放入 generationConfig，这才是 Gemini API 的标准做法
        safe_seed = int(seed_value)
        if safe_seed < 0:
            safe_seed = 0
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647
        payload["generationConfig"] = {"seed": safe_seed}

        if 谷歌搜索:
            payload["tools"] = [{"googleSearch": {}}]

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        print(f"[ComfyUI-shaobkj] Sending inference request to {base_url} (Model: {model})...")
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

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
            api_resp_text = json.dumps(res_json, ensure_ascii=False)
            if not isinstance(api_resp_text, str):
                api_resp_text = str(api_resp_text)
            if len(api_resp_text) > 8000:
                api_resp_text = api_resp_text[:8000] + "...(truncated)"
            return (generated_text, api_resp_text)
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[ComfyUI-shaobkj] Inference Error: {error_msg}")
            raise RuntimeError(f"Inference Failed: {str(e)}") from e
