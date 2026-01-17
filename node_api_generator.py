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
from urllib.parse import urlparse

from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    pil_to_tensor,
    post_json_with_retry,
    resize_pil_long_side,
    tensor_to_pil,
)
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
                "模型选择": (
                    [
                        "gemini-3-pro-image-preview",
                    ],
                    {"default": "gemini-3-pro-image-preview"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "分辨率": (["1k", "2k", "4k"], {"default": "1k"}),
                "图片比例": (
                    ["原图1比例", "Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],
                    {"default": "原图1比例"},
                ),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
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

    def get_target_size(self, resolution, aspect_ratio):
        target_map = {"1k": 1024, "2k": 2048, "4k": 4096}
        target = target_map.get(str(resolution).lower(), 1024)

        ar = str(aspect_ratio or "Free")
        if ar == "Free":
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

    def generate_image(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词, 图片比例, 长边设置, 等待时间, seed, **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        resolution = 分辨率
        prompt = 提示词
        aspect_ratio = 图片比例
        long_side = 长边设置
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        seed_value = seed

        temperature = 0.7

        if not api_key:
            raise ValueError("API Key is required.")

        model = 模型选择

        if aspect_ratio == "原图1比例":
            img1 = kwargs.get("image_1")
            if img1 is None:
                img1 = kwargs.get("参考图1")
            if img1 is not None:
                pil_img = tensor_to_pil(img1)
                w, h = pil_img.size
                if w and h:
                    target_ratio = w / float(h)
                    ratio_map = {
                        "1:1": 1.0,
                        "16:9": 16.0 / 9.0,
                        "9:16": 9.0 / 16.0,
                        "4:3": 4.0 / 3.0,
                        "3:4": 3.0 / 4.0,
                        "3:2": 3.0 / 2.0,
                        "2:3": 2.0 / 3.0,
                        "21:9": 21.0 / 9.0,
                        "9:21": 9.0 / 21.0,
                    }
                    aspect_ratio = min(ratio_map.keys(), key=lambda k: abs(ratio_map[k] - target_ratio))
            else:
                aspect_ratio = "Free"

        input_images = []
        for i in range(1, 50):
            img_key_new = f"image_{i}"
            if img_key_new in kwargs and kwargs[img_key_new] is not None:
                input_images.append(kwargs[img_key_new])
            img_key_old = f"参考图{i}"
            if img_key_old in kwargs and kwargs[img_key_old] is not None:
                input_images.append(kwargs[img_key_old])

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        url = f"{base_origin}/v1beta/models/{model}:generateContent"

        parts = [{"text": prompt}]
        for img_tensor in input_images:
            tensors = [img_tensor]
            if isinstance(img_tensor, torch.Tensor) and img_tensor.dim() == 4:
                tensors = [img_tensor[i] for i in range(img_tensor.shape[0])]
            for t in tensors:
                pil_img = tensor_to_pil(t)
                pil_img = resize_pil_long_side(pil_img, long_side)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                parts.append({"inline_data": {"mime_type": "image/png", "data": img_str}})

        payload = {"contents": [{"role": "user", "parts": parts}]}
        safe_seed = int(seed_value)
        if safe_seed < 0:
            safe_seed = random.randint(0, 2147483647)
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647

        payload["generationConfig"] = {"temperature": temperature, "seed": safe_seed, "responseModalities": ["TEXT", "IMAGE"]}
        payload["generationConfig"]["imageConfig"] = {"imageSize": str(resolution).upper()}
        if aspect_ratio and aspect_ratio != "Free":
            payload["generationConfig"]["imageConfig"]["aspectRatio"] = str(aspect_ratio)

        print(f"[ComfyUI-shaobkj] Sending request to {url} with model {model}...")
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        task_id = None

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

        def try_extract_image_from_json(res_json):
            if isinstance(res_json, dict) and "candidates" in res_json and isinstance(res_json["candidates"], list) and res_json["candidates"]:
                for cand in res_json["candidates"]:
                    content = cand.get("content") if isinstance(cand, dict) else None
                    parts = content.get("parts") if isinstance(content, dict) else None
                    if not isinstance(parts, list):
                        continue
                    for part in parts:
                        if not isinstance(part, dict):
                            continue
                        inline = part.get("inlineData") or part.get("inline_data")
                        if isinstance(inline, dict) and inline.get("data"):
                            image_data = base64.b64decode(inline["data"])
                            image = Image.open(io.BytesIO(image_data))
                            if image.mode != "RGB":
                                image = image.convert("RGB")
                            return pil_to_tensor(image), format_basic_api_response("成功", pil_image=image), image

            if isinstance(res_json, dict) and "data" in res_json and isinstance(res_json["data"], list) and res_json["data"]:
                data_item = res_json["data"][0]
                if isinstance(data_item, dict) and "b64_json" in data_item:
                    image_data = base64.b64decode(data_item["b64_json"])
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return pil_to_tensor(image), format_basic_api_response("成功", pil_image=image), image
                if isinstance(data_item, dict) and "url" in data_item:
                    image_url = data_item["url"]
                    download_timeout = 60 if timeout_val is None else timeout_val
                    img_headers = auth_headers_for_same_origin(str(image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                    img_res = session.get(image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                    img_res.raise_for_status()
                    image = Image.open(io.BytesIO(img_res.content))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return pil_to_tensor(image), format_basic_api_response("成功", pil_image=image), image

            if isinstance(res_json, dict) and "choices" in res_json and isinstance(res_json["choices"], list) and len(res_json["choices"]) > 0:
                content_text = res_json["choices"][0].get("message", {}).get("content", "")
                if content_text is None:
                    content_text = ""

                urls = re.findall(r"!\[.*?\]\((.*?)\)", content_text)
                if not urls:
                    urls = re.findall(r"(https?://[^\s)]+)", content_text)

                valid_image_url = None
                for u in urls:
                    if str(u).lower().startswith("data:"):
                        continue
                    valid_image_url = u
                    break

                if valid_image_url:
                    try:
                        download_timeout = 60 if timeout_val is None else timeout_val
                        img_headers = auth_headers_for_same_origin(str(valid_image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                        img_res = session.get(valid_image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                        img_res.raise_for_status()
                        image = Image.open(io.BytesIO(img_res.content))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        return pil_to_tensor(image), format_basic_api_response("成功", pil_image=image), image
                    except Exception:
                        pass

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
                        image_data = base64.b64decode(b64_clean)
                        image = Image.open(io.BytesIO(image_data))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        return pil_to_tensor(image), format_basic_api_response("成功", pil_image=image), image
                except Exception:
                    pass

            return None

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
            pbar.update_absolute(50)

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

            extracted = try_extract_image_from_json(res_json)
            if extracted:
                img_tensor, raw_text, pil_image = extracted
                return return_result(img_tensor, raw_text, pil_image=pil_image)

            if isinstance(res_json, dict):
                task_id = res_json.get("id") or res_json.get("task_id")
                if not task_id and "data" in res_json and isinstance(res_json["data"], dict):
                    task_id = res_json["data"].get("id") or res_json["data"].get("task_id")
            if task_id:
                print(f"[ComfyUI-shaobkj] 任务ID: {task_id}. 开始轮询状态...")
                poll_url = f"{url}/{task_id}"
                poll_timeout_val = 86400 if int(等待时间) == 0 else int(等待时间)
                start_time = time.time()
                current_p = 70
                fail_count = 0
                done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
                failed_statuses = {"FAILED", "FAIL", "ERROR", "CANCELED", "CANCELLED"}

                while True:
                    if time.time() - start_time > poll_timeout_val:
                        raise RuntimeError(f"图像生成超时 ({poll_timeout_val}秒)")

                    time.sleep(5)
                    current_p = min(95, current_p + 2)
                    pbar.update_absolute(current_p)

                    try:
                        poll_resp = session.get(
                            poll_url,
                            headers=headers,
                            params={"_t": int(time.time() * 1000)},
                            verify=False,
                            timeout=30,
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
                    extracted = try_extract_image_from_json(poll_json)
                    if extracted:
                        img_tensor, raw_text, pil_image = extracted
                        return return_result(img_tensor, raw_text, pil_image=pil_image)

                    status = None
                    if isinstance(poll_json, dict):
                        status = poll_json.get("status") or poll_json.get("task_status")
                        if not status and "data" in poll_json and isinstance(poll_json["data"], dict):
                            status = poll_json["data"].get("status") or poll_json["data"].get("task_status")
                    status_str = str(status).strip().upper() if status is not None else ""
                    if status_str in failed_statuses:
                        raise RuntimeError(f"图像生成失败: {json.dumps(poll_json, indent=2, ensure_ascii=False)}")
                    if status_str in done_statuses:
                        raise RuntimeError(f"任务已完成但未找到图像: {json.dumps(poll_json, indent=2, ensure_ascii=False)}")

            if (not input_images) and isinstance(model, str) and ("image" in model.lower()):
                img_url = f"{base_url}/images/generations"
                img_payload = {
                    "model": model,
                    "prompt": prompt,
                    "n": 1,
                    "size": f"{width}x{height}",
                    "response_format": "b64_json",
                    "user": "comfyui-shaobkj-user",
                    "seed": seed_value,
                }
                img_resp = post_json_with_retry(
                    session,
                    img_url,
                    headers=headers,
                    payload=img_payload,
                    timeout=submit_timeout,
                    proxies=proxies,
                    verify=False,
                )
                if img_resp.status_code not in (200, 201, 202):
                    try:
                        err_json = img_resp.json()
                    except Exception:
                        err_json = img_resp.text
                    raise RuntimeError(f"API Error {img_resp.status_code}: {err_json}")

                img_json = img_resp.json()
                extracted = try_extract_image_from_json(img_json)
                if extracted:
                    img_tensor, raw_text, pil_image = extracted
                    return return_result(img_tensor, raw_text, pil_image=pil_image)

            raise RuntimeError(f"No image found in API response. Response: {json.dumps(res_json, indent=2, ensure_ascii=False)}")
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[ComfyUI-shaobkj] {error_msg}")
            raise RuntimeError(f"Generation Failed: {str(e)}") from e
