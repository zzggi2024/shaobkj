import base64
import io
import json
import os
import random
import re
import time
import traceback
from urllib.parse import urlparse

import folder_paths
import requests
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl

from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    crop_image_to_ratio,
    disable_insecure_request_warnings,
    extract_task_id_and_video_url,
    get_config_value,
    resize_pil_long_side,
    robust_download_video,
    sanitize_text,
    tensor_to_pil,
)


def _encode_image_to_data_url(image_tensor, long_side=1280, aspect_ratio=None):
    if image_tensor is None:
        return None
    pil_img = tensor_to_pil(image_tensor)
    if aspect_ratio:
        pil_img = crop_image_to_ratio(pil_img, str(aspect_ratio), bbox=(0, 0, pil_img.size[0], pil_img.size[1]))
    pil_img = resize_pil_long_side(pil_img, int(long_side))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _encode_image_to_png_bytes(image_tensor, long_side=1280, aspect_ratio=None):
    if image_tensor is None:
        return None
    pil_img = tensor_to_pil(image_tensor)
    if aspect_ratio:
        pil_img = crop_image_to_ratio(pil_img, str(aspect_ratio), bbox=(0, 0, pil_img.size[0], pil_img.size[1]))
    pil_img = resize_pil_long_side(pil_img, int(long_side))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return buffered.getvalue()


class Shaobkj_Grok_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型": (
                    ["grok-imagine-1.0-video", "grok-imagine-1.0-video-20s", "grok-imagine-1.0-video-30s"],
                    {"default": "grok-imagine-1.0-video", "tooltip": "Grok 视频模型；20s/30s 模型将按模型名自动锁定时长"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "提示词": ("STRING", {"multiline": True, "default": "让主体自然动起来，增加轻微运镜和环境动态", "tooltip": "视频内容描述；推荐：简洁具体"}),
                "分辨率": (["HD", "SD"], {"default": "HD", "tooltip": "video_config.resolution；推荐：HD(720p) 或 SD(480p)"}),
                "画幅比例": (["16:9", "9:16", "1:1", "2:3", "3:2"], {"default": "16:9", "tooltip": "video_config.aspect_ratio；推荐：16:9 / 9:16"}),
                "生成时长": (["6", "10", "15", "20", "30"], {"default": "6", "tooltip": "生成时长需与模型匹配；20/30 秒模型将自动锁定时长"}),
                "预设": (["normal"], {"default": "normal", "tooltip": "video_config.preset；当前文档示例为 normal"}),
                "流式": ("BOOLEAN", {"default": False, "tooltip": "stream；文档示例推荐 false"}),
                "输入图像_长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0 为无限等待；推荐：0"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": "randomize", "tooltip": "随机种子；0 为不传"}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False, "tooltip": "API 申请入口；推荐：默认地址"}),
            },
            "optional": {
                "参考图1": ("IMAGE", {"tooltip": "可选参考图 1；连接后将按 OpenAI content.image_url 方式发送"}),
                "参考图2": ("IMAGE", {"tooltip": "可选参考图 2；可继续自动扩展"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "任务ID", "API响应", "视频链接")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_video(
        self,
        API密钥,
        API地址,
        模型,
        使用系统代理,
        提示词,
        分辨率,
        画幅比例,
        生成时长,
        预设,
        流式,
        输入图像_长边设置,
        等待时间,
        seed,
        API申请地址,
        参考图1=None,
        参考图2=None,
        **kwargs,
    ):
        if not str(API密钥).strip():
            raise ValueError("API Key is required.")

        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        base_url = str(API地址).rstrip("/")
        root_base = base_url[:-3] if base_url.endswith("/v1") else base_url
        submit_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        submit_url = f"{submit_base}/chat/completions"
        api_origin = urlparse(root_base).netloc

        final_duration = int(生成时长)
        model_name = str(模型).strip()
        if model_name.endswith("-20s"):
            final_duration = 20
        elif model_name.endswith("-30s"):
            final_duration = 30
        elif final_duration not in {6, 10, 15}:
            final_duration = 6

        aspect_ratio_value = str(画幅比例).strip()
        resolution_value = "720P" if str(分辨率).strip().upper() == "HD" else "480P"

        reference_images = []
        for value in (参考图1, 参考图2):
            if value is not None:
                reference_images.append(value)
        dynamic_ref_pairs = []
        for key, value in kwargs.items():
            key_str = str(key)
            if key_str.startswith("参考图") and key_str[3:].isdigit() and value is not None:
                dynamic_ref_pairs.append((int(key_str[3:]), value))
        dynamic_ref_pairs.sort(key=lambda item: item[0])
        for _, value in dynamic_ref_pairs:
            reference_images.append(value)

        image_refs = []
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "close",
            "Authorization": f"Bearer {str(API密钥).strip()}",
        }

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

        for image_tensor in reference_images:
            image_data_url = _encode_image_to_data_url(
                image_tensor,
                int(输入图像_长边设置),
            )
            if image_data_url:
                image_refs.append(image_data_url)

        openai_content = []
        for image_data_url in image_refs:
            openai_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        openai_content.append({"type": "text", "text": str(提示词)})

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": openai_content}],
            "stream": bool(流式),
            "video_config": {
                "duration": final_duration,
                "aspect_ratio": aspect_ratio_value,
                "resolution": str(分辨率).strip().upper(),
                "preset": str(预设).strip() or "normal",
            },
            "duration": final_duration,
            "aspect_ratio": aspect_ratio_value,
            "resolution": str(分辨率).strip().upper(),
        }
        if int(seed) > 0:
            payload["seed"] = int(seed)
        print(
            f"[Shaobkj-Grok] submit model={model_name}, aspect_ratio={aspect_ratio_value}, "
            f"resolution={resolution_value}, duration={final_duration}, refs={len(image_refs)}"
        )
        print(f"[Shaobkj-Grok] submit payload={sanitize_text(json.dumps(payload, ensure_ascii=False), max_len=2000)}")

        try:
            pbar.update_absolute(10)
            resp = session.post(
                submit_url,
                headers=headers,
                json=payload,
                timeout=submit_timeout,
                proxies=proxies,
                verify=False,
            )
        except Exception as e:
            error_msg = f"Connection Failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[Shaobkj-Grok] {error_msg}")
            raise RuntimeError(f"Connection Failed: {str(e)}") from e

        if resp.status_code not in (200, 201, 202):
            try:
                err_msg = resp.json()
            except Exception:
                err_msg = resp.text
            print(f"[Shaobkj-Grok] API Error {resp.status_code}: {err_msg}")
            raise RuntimeError(f"API Error {resp.status_code}: {err_msg}")

        pbar.update_absolute(30)
        task_id, direct_video_url, parsed_json, raw_text = extract_task_id_and_video_url(resp)
        resp_json = parsed_json if parsed_json is not None else {"raw": raw_text}
        if not task_id:
            task_id = self._find_task_id(resp_json)
        print(f"[Shaobkj-Grok] submit response={sanitize_text(self._stringify_payload(resp_json), max_len=2000)}")

        direct_video_url = direct_video_url or self._find_video_url(resp_json)
        if direct_video_url:
            pbar.update_absolute(90)
            video_obj = self._download_video(
                direct_video_url,
                session=session,
                headers=headers,
                api_origin=api_origin,
                timeout_budget=None if int(等待时间) == 0 else int(等待时间),
            )
            pbar.update_absolute(100)
            return (video_obj, str(task_id or ""), self._stringify_payload(resp_json), direct_video_url)

        if not task_id:
            print(f"[Shaobkj-Grok] 未找到任务ID或视频链接详情: {self._stringify_payload(resp_json)}")
            raise RuntimeError("未找到任务ID (id/task_id) 或视频链接")

        timeout_val = 86400 if int(等待时间) == 0 else int(等待时间)
        start_time = time.time()
        poll_interval = 5
        fail_count = 0
        pbar.update_absolute(40)

        poll_candidates = [
            (f"{root_base}/v1/videos/{task_id}", {"model": model_name}),
            (f"{root_base}/v1/video/generations/{task_id}", None),
            (f"{root_base}/v2/videos/generations/{task_id}", None),
        ]
        content_candidates = [
            f"{root_base}/v1/videos/{task_id}/content",
            f"{root_base}/v1/video/generations/{task_id}/content",
            f"{root_base}/v2/videos/generations/{task_id}/content",
        ]

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_val - elapsed
            if remaining <= 0:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")

            time.sleep(min(poll_interval, max(0.0, remaining)))
            poll_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))

            poll_resp = None
            for poll_url, poll_json_body in poll_candidates:
                try:
                    if poll_json_body is not None:
                        current_resp = session.get(
                            poll_url,
                            headers=headers,
                            json=poll_json_body,
                            params={"_t": int(time.time() * 1000)},
                            verify=False,
                            timeout=poll_timeout,
                            proxies=proxies,
                        )
                    else:
                        current_resp = session.get(
                            poll_url,
                            headers=headers,
                            params={"_t": int(time.time() * 1000)},
                            verify=False,
                            timeout=poll_timeout,
                            proxies=proxies,
                        )
                    if current_resp.status_code == 200:
                        poll_resp = current_resp
                        break
                except Exception:
                    continue

            if poll_resp is None:
                fail_count += 1
                if fail_count >= 10:
                    raise RuntimeError("轮询连续失败 10 次")
                continue

            fail_count = 0
            try:
                poll_json = poll_resp.json()
            except Exception:
                poll_json = {"raw": poll_resp.text}

            status = self._find_status(poll_json)
            progress_val = self._find_progress(poll_json)
            video_url = self._find_video_url(poll_json)
            error_message = self._find_error_message(poll_json)

            if status in {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}:
                print(f"[Shaobkj-Grok] 任务失败详情: {self._stringify_payload(poll_json)}")
                if error_message:
                    raise RuntimeError(f"任务失败，{error_message}")
                raise RuntimeError("任务失败")

            if not video_url and status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"}:
                for content_url in content_candidates:
                    try:
                        content_resp = session.get(
                            content_url,
                            headers=headers,
                            params={"_t": int(time.time() * 1000)},
                            verify=False,
                            timeout=poll_timeout,
                            proxies=proxies,
                            stream=True,
                        )
                        if content_resp.status_code == 200:
                            pbar.update_absolute(92)
                            dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                            video_obj = self._download_video(
                                content_url,
                                session=session,
                                headers=headers,
                                api_origin=api_origin,
                                timeout_budget=dl_budget,
                            )
                            pbar.update_absolute(100)
                            return (video_obj, str(task_id), self._stringify_payload(poll_json), str(content_url))
                    except Exception:
                        continue

            if video_url:
                pbar.update_absolute(92)
                dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                video_obj = self._download_video(
                    video_url,
                    session=session,
                    headers=headers,
                    api_origin=api_origin,
                    timeout_budget=dl_budget,
                )
                pbar.update_absolute(100)
                return (video_obj, str(task_id), self._stringify_payload(poll_json), str(video_url))

            if progress_val is not None:
                pbar.update_absolute(min(90, 40 + int(float(progress_val) * 0.5)))
            else:
                guessed = min(88, 40 + int((elapsed / max(1, timeout_val)) * 48))
                pbar.update_absolute(guessed)

    @staticmethod
    def _stringify_payload(payload):
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload)

    @staticmethod
    def _find_task_id(obj):
        if isinstance(obj, dict):
            for key in ("task_id", "id", "taskId", "video_id", "generation_id"):
                value = obj.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
            for key in ("data", "result", "output", "task", "video"):
                if key in obj:
                    found = Shaobkj_Grok_Video._find_task_id(obj.get(key))
                    if found:
                        return found
            for value in obj.values():
                found = Shaobkj_Grok_Video._find_task_id(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok_Video._find_task_id(item)
                if found:
                    return found
        return ""

    @staticmethod
    def _find_status(obj):
        if isinstance(obj, dict):
            for key in ("status", "task_status", "state"):
                value = obj.get(key)
                if value is not None:
                    return str(value).strip().upper()
            for value in obj.values():
                found = Shaobkj_Grok_Video._find_status(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok_Video._find_status(item)
                if found:
                    return found
        return ""

    @staticmethod
    def _find_progress(obj):
        if isinstance(obj, dict):
            for key in ("progress", "percentage"):
                if key in obj:
                    try:
                        value = float(str(obj[key]).replace("%", "").strip())
                        if value <= 1.0:
                            value *= 100.0
                        return max(0.0, min(100.0, value))
                    except Exception:
                        pass
            for value in obj.values():
                found = Shaobkj_Grok_Video._find_progress(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok_Video._find_progress(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _find_error_message(obj):
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("code")
                if msg:
                    return str(msg)
            if isinstance(err, str) and err.strip():
                return err.strip()
            for key in ("message", "fail_reason", "reason"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            for value in obj.values():
                found = Shaobkj_Grok_Video._find_error_message(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok_Video._find_error_message(item)
                if found:
                    return found
        return None

    @staticmethod
    def _is_video_url(text):
        clean = str(text or "").split("?", 1)[0].lower().rstrip(".,;!?")
        return clean.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"))

    @staticmethod
    def _is_image_url(text):
        clean = str(text or "").split("?", 1)[0].lower().rstrip(".,;!?")
        return clean.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"))

    @staticmethod
    def _find_video_url(obj):
        candidates = Shaobkj_Grok_Video._collect_urls(obj)
        for url in candidates:
            if Shaobkj_Grok_Video._is_video_url(url):
                return url
        for url in candidates:
            if not Shaobkj_Grok_Video._is_image_url(url):
                return url
        return None

    @staticmethod
    def _collect_urls(obj):
        urls = []
        if isinstance(obj, str):
            for match in re.finditer(r"(https?://[^\s\]\)\"']+)", obj):
                url = match.group(1).rstrip(".,;!?")
                if url:
                    urls.append(url)
        elif isinstance(obj, dict):
            for key in ("video_url", "content_url", "download_url", "output", "url"):
                value = obj.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    urls.append(value.rstrip(".,;!?"))
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        urls.extend(Shaobkj_Grok_Video._collect_urls(choice.get("message")))
            for key in ("content", "message", "messages", "data", "result", "output"):
                if key in obj:
                    urls.extend(Shaobkj_Grok_Video._collect_urls(obj.get(key)))
            for value in obj.values():
                urls.extend(Shaobkj_Grok_Video._collect_urls(value))
        elif isinstance(obj, list):
            for item in obj:
                urls.extend(Shaobkj_Grok_Video._collect_urls(item))

        deduped = []
        seen = set()
        for url in urls:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        return deduped

    @staticmethod
    def _download_video(video_url, session=None, headers=None, api_origin="", timeout_budget=None):
        dl_headers = auth_headers_for_same_origin(str(video_url), api_origin, headers or {})
        dl_timeout = 300 if timeout_budget is None else max(30, int(timeout_budget))
        filename = f"grok_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
        output_dir = folder_paths.get_output_directory()
        file_path = os.path.join(output_dir, filename)

        print(f"[Shaobkj-Grok] downloading video: {video_url}")
        success = robust_download_video(video_url, file_path, max_retries=3, timeout=dl_timeout, headers=dl_headers)
        if not success:
            print("[Shaobkj-Grok] robust download failed, trying requests fallback")
            if session:
                resp = session.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            else:
                resp = requests.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            resp.raise_for_status()
            content_type = str(resp.headers.get("Content-Type", "")).lower()
            if content_type and not content_type.startswith("video/") and "octet-stream" not in content_type:
                raise RuntimeError(f"下载链接不是视频文件: Content-Type={content_type}, URL={video_url}")
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        if file_size <= 0:
            raise RuntimeError("视频下载失败：本地文件为空")
        print(f"[Shaobkj-Grok] video saved: {file_path} ({file_size} bytes)")
        return InputImpl.VideoFromFile(file_path)
