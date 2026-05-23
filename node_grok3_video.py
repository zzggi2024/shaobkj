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
    disable_insecure_request_warnings,
    get_config_value,
    resize_pil_long_side,
    sanitize_text,
    tensor_to_pil,
)


def _encode_image_to_data_url(image_tensor, long_side=1280):
    if image_tensor is None:
        return None
    pil_img = resize_pil_long_side(tensor_to_pil(image_tensor), int(long_side))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


class Shaobkj_Grok3_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yunwu.ai", "multiline": False, "tooltip": "云雾视频 API 地址；会自动请求 /v1/video/create；推荐：https://yunwu.ai"}),
                "模型": (["grok-video-3", "grok-video-3-10s", "自定义"], {"default": "grok-video-3", "tooltip": "视频模型；默认：grok-video-3"}),
                "自定义模型名称": ("STRING", {"default": "", "multiline": False, "tooltip": "当‘模型’选择‘自定义’时生效；请输入实际模型名称"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "提示词": ("STRING", {"multiline": True, "default": "让主体自然动起来，增加轻微运镜和环境动态", "tooltip": "视频内容描述；推荐：简洁具体"}),
                "画幅比例": (["2:3", "3:2", "16:9", "9:16", "1:1"], {"default": "3:2", "tooltip": "视频画幅比例；文档示例可选 2:3 / 3:2 / 1:1"}),
                "生成时长": (["6", "10", "15"], {"default": "10", "tooltip": "保留字段；当前这份接口文档不提交时长参数"}),
                "分辨率": (["720P", "1080P"], {"default": "720P", "tooltip": "接口 size 字段；文档说明当前暂只支持 720P"}),
                "输入图像_长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0 为无限等待；推荐：0"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "保留字段；当前接口文档未使用"}),
                "API申请地址": ("STRING", {"default": "https://yunwu.ai", "multiline": False, "tooltip": "API 申请入口；推荐：默认地址"}),
            },
            "optional": {
                "参考图1": ("IMAGE", {"tooltip": "可选参考图 1；会转为 data URL 写入 images"}),
                "参考图2": ("IMAGE", {"tooltip": "可选参考图 2；可继续自动扩展"}),

            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "任务ID", "API响应", "视频链接")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APlbox"

    def generate_video(
        self,
        API密钥,
        API地址,
        模型,
        自定义模型名称,
        使用系统代理,
        提示词,
        画幅比例,
        生成时长,
        分辨率,
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

        base_url = str(API地址 or "https://yunwu.ai").rstrip("/")
        root_base = base_url[:-3] if base_url.endswith("/v1") else base_url
        submit_base = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        submit_url = f"{submit_base}/video/create"

        api_origin = urlparse(root_base).netloc

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {str(API密钥).strip()}",
        }

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

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
        reference_images = reference_images[:7]

        image_urls = []
        for idx, image_tensor in enumerate(reference_images, start=1):
            pbar.update_absolute(min(25, 12 + idx * 2))
            image_data_url = _encode_image_to_data_url(image_tensor, int(输入图像_长边设置))
            if not image_data_url:
                raise RuntimeError(f"参考图{idx}编码失败")
            image_urls.append(image_data_url)

        if not image_urls:
            raise RuntimeError("至少需要连接一张参考图")

        model_name = str(自定义模型名称).strip() if str(模型).strip() == "自定义" else str(模型).strip()
        if not model_name:
            raise ValueError("请选择模型或填写自定义模型名称。")
        aspect_ratio_value = str(画幅比例).strip()
        resolution_value = str(分辨率).strip().upper()
        prompt_text = str(提示词).strip()

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "aspect_ratio": aspect_ratio_value,
            "size": resolution_value,
            "images": image_urls,
        }

        log_payload = dict(payload)
        log_payload["images"] = [sanitize_text(url, max_len=120) for url in image_urls]

        print(f"[Shaobkj-Grok3] final model={model_name}")
        print(
            f"[Shaobkj-Grok3] submit model={model_name}, aspect_ratio={aspect_ratio_value}, "
            f"size={resolution_value}, refs={len(image_urls)}"
        )

        print(f"[Shaobkj-Grok3] submit payload={sanitize_text(json.dumps(log_payload, ensure_ascii=False), max_len=2000)}")

        try:
            pbar.update_absolute(30)
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
            print(f"[Shaobkj-Grok3] {error_msg}")
            raise RuntimeError(f"Connection Failed: {str(e)}") from e

        if resp.status_code != 200:
            print(f"[Shaobkj-Grok3] API Error {resp.status_code}: {resp.text}")
            raise RuntimeError(
                f"API Error {resp.status_code}: model={model_name}, url={submit_url}, response={resp.text}"
            )

        try:
            submit_payload = resp.json()
        except Exception as e:
            raise RuntimeError(f"提交响应不是JSON，请确认 API地址 的 /v1/video/create 接口可用：{resp.text[:1000]}") from e

        print(f"[Shaobkj-Grok3] submit response={sanitize_text(self._stringify_payload(submit_payload), max_len=2000)}")
        video_url = self._find_video_url(submit_payload)
        task_id = str(self._find_task_id(submit_payload) or "").strip()
        if video_url:
            video_obj = self._download_video(
                video_url,
                session=session,
                headers=headers,
                api_origin=api_origin,
                timeout_budget=None if int(等待时间) == 0 else int(等待时间),
            )
            pbar.update_absolute(100)
            return (
                video_obj,
                task_id,
                self._stringify_payload(submit_payload),
                video_url,
            )

        if not task_id:
            raise RuntimeError("未找到任务ID或视频链接")

        timeout_val = 600 if int(等待时间) == 0 else int(等待时间)
        start_time = time.time()
        attempts = 0
        max_attempts = 200
        pbar.update_absolute(40)
        poll_candidates = [
            f"{submit_base}/video/query?id={task_id}",
            f"{root_base}/v1/video/query?id={task_id}",
            f"{submit_base}/video/query/{task_id}",
            f"{submit_base}/videos/{task_id}",
            f"{root_base}/v1/video/query/{task_id}",
            f"{root_base}/v1/videos/{task_id}",
        ]



        while attempts < max_attempts:
            elapsed = time.time() - start_time
            if elapsed > timeout_val:
                raise RuntimeError(f"视频生成超时 ({elapsed:.1f}秒)")

            time.sleep(5)
            attempts += 1
            poll_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(timeout_val - elapsed)))

            poll_payload = None
            for poll_url in poll_candidates:
                try:
                    poll_resp = session.get(
                        poll_url,
                        headers=headers,
                        timeout=poll_timeout,
                        proxies=proxies,
                        verify=False,
                    )
                    if poll_resp.status_code == 200:
                        poll_payload = poll_resp.json()
                        break
                except requests.exceptions.Timeout:
                    continue
                except Exception:
                    continue
            if poll_payload is None:
                continue

            status = str(self._find_status(poll_payload) or "UNKNOWN").upper()
            progress = self._find_progress(poll_payload)
            video_url = self._find_video_url(poll_payload)
            error_message = self._find_error_message(poll_payload)

            if video_url:
                dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                video_obj = self._download_video(
                    video_url,
                    session=session,
                    headers=headers,
                    api_origin=api_origin,
                    timeout_budget=dl_budget,
                )
                pbar.update_absolute(100)
                return (
                    video_obj,
                    task_id,
                    self._stringify_payload(poll_payload),
                    video_url,
                )

            if status in {"FAILURE", "FAILED", "FAIL", "ERROR", "CANCELED", "CANCELLED"}:
                raise RuntimeError(f"Video generation failed: {error_message or 'Unknown error'}")

            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"}:
                fallback_candidates = self._collect_urls(poll_payload)
                for candidate_url in fallback_candidates:
                    if not self._is_image_url(candidate_url):
                        try:
                            dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                            video_obj = self._download_video(
                                candidate_url,
                                session=session,
                                headers=headers,
                                api_origin=api_origin,
                                timeout_budget=dl_budget,
                            )
                            pbar.update_absolute(100)
                            return (
                                video_obj,
                                task_id,
                                self._stringify_payload(poll_payload),
                                candidate_url,
                            )
                        except Exception:
                            continue

            if progress is not None:
                pbar.update_absolute(min(90, 40 + int(float(progress) * 0.5)))
            else:
                pbar.update_absolute(min(80, 40 + attempts * 40 // max_attempts))

        raise RuntimeError(f"Video generation timeout or failed to retrieve video URL after {attempts} attempts")

    @staticmethod
    def _find_task_id(obj):
        if isinstance(obj, dict):
            for key in ("task_id", "id", "taskId", "video_id", "generation_id"):
                value = obj.get(key)
                if value is not None:
                    text = str(value).strip()
                    if text:
                        return text
            for value in obj.values():
                found = Shaobkj_Grok3_Video._find_task_id(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok3_Video._find_task_id(item)
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
                found = Shaobkj_Grok3_Video._find_status(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok3_Video._find_status(item)
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
                found = Shaobkj_Grok3_Video._find_progress(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok3_Video._find_progress(item)
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
                found = Shaobkj_Grok3_Video._find_error_message(value)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = Shaobkj_Grok3_Video._find_error_message(item)
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
        candidates = Shaobkj_Grok3_Video._collect_urls(obj)
        for url in candidates:
            if Shaobkj_Grok3_Video._is_video_url(url):
                return url
        for url in candidates:
            if not Shaobkj_Grok3_Video._is_image_url(url):
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
            priority_keys = ("video_url", "content_url", "download_url", "output", "url")
            for key in priority_keys:
                value = obj.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    urls.append(value.rstrip(".,;!?"))
            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        urls.extend(Shaobkj_Grok3_Video._collect_urls(choice.get("message")))
            for key in ("content", "message", "messages", "data", "result", "output"):
                if key in obj:
                    urls.extend(Shaobkj_Grok3_Video._collect_urls(obj.get(key)))
            for value in obj.values():
                urls.extend(Shaobkj_Grok3_Video._collect_urls(value))
        elif isinstance(obj, list):
            for item in obj:
                urls.extend(Shaobkj_Grok3_Video._collect_urls(item))

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
        dl_timeout = 300 if timeout_budget is None else max(1, int(timeout_budget))
        filename = f"grok3_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
        output_dir = folder_paths.get_output_directory()
        file_path = os.path.join(output_dir, filename)

        print(f"[Shaobkj-Grok3] downloading video: {video_url}")
        if session:
            if dl_headers:
                resp = session.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            else:
                resp = session.get(video_url, stream=True, verify=False, timeout=dl_timeout)
        else:
            if dl_headers:
                resp = requests.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            else:
                resp = requests.get(video_url, stream=True, verify=False, timeout=dl_timeout)
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
        print(f"[Shaobkj-Grok3] video saved: {file_path} ({file_size} bytes)")
        return InputImpl.VideoFromFile(file_path)

    @staticmethod
    def _stringify_payload(payload):
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return str(payload)

