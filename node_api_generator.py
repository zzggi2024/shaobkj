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

import torch.nn.functional as F

from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    pil_to_tensor,
    post_json_with_retry,
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
                    ["Free", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],
                    {"default": "Free"},
                ),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
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

    def generate_image(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词, 图片比例, 等待时间, seed, **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        resolution = 分辨率
        prompt = 提示词
        aspect_ratio = 图片比例
        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        seed_value = seed

        temperature = 0.7

        if not api_key:
            raise ValueError("API Key is required.")

        model = 模型选择

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

        url = f"{base_origin}/v1beta/models/{model}:generateContent"

        parts = [{"text": prompt}]

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
                failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}

                while True:
                    elapsed = time.time() - start_time
                    remaining = poll_timeout_val - elapsed
                    if remaining <= 0:
                        raise RuntimeError(f"图像生成超时 ({poll_timeout_val}秒)")

                    time.sleep(min(5, max(0.0, remaining)))
                    current_p = min(95, current_p + 2)
                    pbar.update_absolute(current_p)

                    try:
                        poll_req_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))
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
                        raise RuntimeError(f"图像生成失败: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")
                    if status_str in done_statuses:
                        raise RuntimeError(f"任务已完成但未找到图像: {sanitize_text(json.dumps(poll_json, ensure_ascii=False))}")

            raise RuntimeError(f"No image found in API response. Response: {sanitize_text(json.dumps(res_json, ensure_ascii=False))}")
        except Exception as e:
            error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"[ComfyUI-shaobkj] {error_msg}")
            raise RuntimeError(f"Generation Failed: {str(e)}") from e


class Shaobkj_APINode_Batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词列表": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "一只猫\n一只狗"}),
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "API响应")
    FUNCTION = "generate_images_batch"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_images_batch(self, API密钥, API地址, 模型选择, 使用系统代理, 分辨率, 提示词列表, 图片比例, 等待时间, 并发数, seed, **kwargs):
        api_key = API密钥
        base_origin = str(API地址).rstrip("/")
        api_origin = urlparse(base_origin).netloc
        model = 模型选择
        resolution = 分辨率
        aspect_ratio = 图片比例
        timeout_val = None if int(等待时间) == 0 else int(等待时间)

        if not api_key:
            raise ValueError("API Key is required.")

        prompts = [p.strip() for p in str(提示词列表 or "").splitlines() if p.strip()]
        if not prompts:
            raise ValueError("提示词列表不能为空。")

        disable_insecure_request_warnings()

        base_headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        submit_timeout = build_submit_timeout(int(等待时间))

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

        def try_extract_image_from_json(res_json, session, proxies):
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
                            return pil_to_tensor(image), image

            if isinstance(res_json, dict) and "data" in res_json and isinstance(res_json["data"], list) and res_json["data"]:
                data_item = res_json["data"][0]
                if isinstance(data_item, dict) and "b64_json" in data_item:
                    image_data = base64.b64decode(data_item["b64_json"])
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return pil_to_tensor(image), image
                if isinstance(data_item, dict) and "url" in data_item:
                    image_url = data_item["url"]
                    download_timeout = 60 if timeout_val is None else timeout_val
                    img_headers = auth_headers_for_same_origin(str(image_url), api_origin, {"Authorization": f"Bearer {api_key}"})
                    img_res = session.get(image_url, verify=False, timeout=download_timeout, proxies=proxies, headers=img_headers)
                    img_res.raise_for_status()
                    image = Image.open(io.BytesIO(img_res.content))
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    return pil_to_tensor(image), image

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
                        return pil_to_tensor(image), image
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
                        return pil_to_tensor(image), image
                except Exception:
                    pass
            return None

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

        def pad_tensor_to(t, max_h, max_w):
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                return t
            b, h, w, c = t.shape
            if h == max_h and w == max_w:
                return t
            tch = t.permute(0, 3, 1, 2)
            pad_w = max_w - w
            pad_h = max_h - h
            padded = F.pad(tch, (0, pad_w, 0, pad_h), "constant", 0)
            return padded.permute(0, 2, 3, 1)

        def generate_one(index, prompt):
            local_seed = normalize_seed(int(seed) + int(index))
            session, proxies = create_requests_session(bool(使用系统代理))
            parts = [{"text": prompt}]
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

            extracted = try_extract_image_from_json(res_json, session, proxies)
            if extracted:
                img_tensor, pil_image = extracted
                return (img_tensor, format_basic_api_response("成功", local_seed, pil_image=pil_image, task_id=task_id))

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
                extracted = try_extract_image_from_json(poll_json, session, proxies)
                if extracted:
                    img_tensor, pil_image = extracted
                    return (img_tensor, format_basic_api_response("成功", local_seed, pil_image=pil_image, task_id=task_id))
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(generate_one, idx, p): (idx, p) for idx, p in enumerate(prompts)}
            for fut in concurrent.futures.as_completed(future_map):
                idx, p = future_map[fut]
                try:
                    img_tensor, resp_text = fut.result()
                    results.append((idx, img_tensor, resp_text))
                except Exception as e:
                    errors.append((idx, sanitize_text(str(e))))

        results.sort(key=lambda x: x[0])
        ok_tensors = [r[1] for r in results if isinstance(r[1], torch.Tensor)]
        if not ok_tensors:
            raise RuntimeError(f"批量生成全部失败，示例错误: {errors[0][1] if errors else '未知错误'}")

        max_h = max(int(t.shape[1]) for t in ok_tensors)
        max_w = max(int(t.shape[2]) for t in ok_tensors)
        padded = [pad_tensor_to(t, max_h, max_w) for t in ok_tensors]
        batch_tensor = torch.cat(padded, dim=0)

        lines = [f"批量生成完成 | 总数: {len(prompts)} | 成功: {len(ok_tensors)} | 失败: {len(errors)}"]
        if errors:
            for idx, msg in errors[:5]:
                snippet = prompts[idx][:30] if idx < len(prompts) else str(idx)
                lines.append(f"失败[{idx}] {snippet}: {sanitize_text(msg)}")
        api_text = "\n".join(lines)
        return {"ui": {"string": [api_text]}, "result": (batch_tensor, api_text)}
