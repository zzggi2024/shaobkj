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
    extract_task_id_and_video_url,
    get_config_value,
    post_with_retry,
    resize_and_encode_image,
    tensor_to_pil,
)


class Shaobkj_SD20_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://wcnb.ai", "multiline": False, "tooltip": "API 基础地址；推荐：https://wcnb.ai"}),
                "模型": ("STRING", {"default": "dance2-fast-15s", "multiline": False, "tooltip": "视频模型名称；推荐：dance2-fast-15s"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "提示词": ("STRING", {"multiline": True, "default": "画面动起来", "tooltip": "视频内容描述；推荐：简洁具体"}),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "参考图长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0 为无限等待；推荐：0"}),
            },
            "optional": {
                "参考图": ("IMAGE", {"tooltip": "可选首帧/参考图；推荐：图生视频时连接"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "API响应")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_video(self, API密钥, API地址, 模型, 使用系统代理, 提示词, 长边设置, 等待时间, 参考图=None, **kwargs):
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        if not API密钥:
            raise ValueError("API Key is required.")

        openai_base = str(API地址).rstrip("/")
        if openai_base.endswith("/v1"):
            openai_base = openai_base[:-3]
        api_url = f"{openai_base}/v1/chat/completions"
        api_origin = urlparse(openai_base).netloc

        content = [{"type": "text", "text": str(提示词)}]
        if 参考图 is not None:
            image_b64, _ = resize_and_encode_image(tensor_to_pil(参考图), int(长边设置))
            if image_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        "role": "reference_image",
                    }
                )

        payload = {
            "model": str(模型).strip(),
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API密钥}",
        }

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

        try:
            pbar.update_absolute(10)
            resp = post_with_retry(
                session,
                api_url,
                headers=headers,
                timeout=submit_timeout,
                proxies=proxies,
                verify=False,
                json=payload,
            )
        except Exception as e:
            error_msg = f"Connection Failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[Shaobkj-SD2.0] {error_msg}")
            raise RuntimeError(f"Connection Failed: {str(e)}") from e

        if resp.status_code not in (200, 201, 202):
            try:
                err_msg = resp.json()
            except Exception:
                err_msg = resp.text
            print(f"[Shaobkj-SD2.0] API Error {resp.status_code}: {err_msg}")
            raise RuntimeError(f"API Error {resp.status_code}: {err_msg}")

        pbar.update_absolute(30)
        task_id, direct_video_url, parsed_json, raw_text = extract_task_id_and_video_url(resp)
        submit_payload = self._to_payload(parsed_json if parsed_json is not None else raw_text)
        content_payload = self._extract_choice_content_payload(parsed_json)

        better_task_id = self._find_task_id(content_payload)
        if better_task_id:
            task_id = better_task_id
        elif self._is_chat_completion_id(task_id):
            task_id = None

        better_video_url = self._find_video_url(content_payload)
        if better_video_url:
            direct_video_url = better_video_url
        if not direct_video_url:
            direct_video_url = self._find_video_url(submit_payload)

        direct_status = self._find_status(content_payload) or self._find_status(submit_payload)
        if direct_video_url and (not task_id or direct_status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}):
            pbar.update_absolute(90)
            return self.download_video(
                direct_video_url,
                submit_payload,
                session=session,
                headers=headers,
                api_origin=api_origin,
                timeout_budget=None if int(等待时间) == 0 else max(1, int(等待时间)),
            )

        if not task_id:
            print(f"[Shaobkj-SD2.0] 未找到任务ID或视频链接详情: {self._stringify_payload(submit_payload)}")
            raise RuntimeError("未找到任务ID (task_id/id) 或视频链接")

        timeout_val = 86400 if int(等待时间) == 0 else int(等待时间)
        start_time = time.time()
        pbar.update_absolute(40)
        poll_interval = 5
        attempts = 0
        done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
        failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}
        poll_urls = [
            f"{openai_base}/v1/chat/completions/{task_id}",
            f"{openai_base}/v1/videos/{task_id}",
            f"{openai_base}/v1/tasks/{task_id}",
        ]
        content_url = f"{openai_base}/v1/videos/{task_id}/content"
        fail_count = 0

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_val - elapsed
            if remaining <= 0:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")

            time.sleep(min(poll_interval, max(0.0, remaining)))
            poll_req_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))

            poll_resp = None
            last_error = None
            for poll_url in poll_urls:
                try:
                    current_resp = session.get(
                        poll_url,
                        headers=headers,
                        params={"_t": int(time.time() * 1000)},
                        verify=False,
                        timeout=poll_req_timeout,
                        proxies=proxies,
                    )
                    if current_resp.status_code in (404, 405):
                        last_error = current_resp
                        continue
                    poll_resp = current_resp
                    break
                except Exception as e:
                    last_error = e

            if poll_resp is None:
                fail_count += 1
                print(f"[Shaobkj-SD2.0] Polling connection error (attempt {fail_count}): {last_error}")
                if fail_count >= 10:
                    raise RuntimeError(f"Polling failed 10 times consecutively. Last error: {last_error}")
                time.sleep(2)
                continue

            attempts += 1
            fail_count = 0

            if poll_resp.status_code != 200:
                try:
                    err_msg = poll_resp.json()
                except Exception:
                    err_msg = poll_resp.text
                raise RuntimeError(f"API Error {poll_resp.status_code}: {err_msg}")

            poll_json = self._to_payload_from_response(poll_resp)
            status_str = self._find_status(poll_json)
            progress_val = self._find_progress(poll_json)
            video_url = self._find_video_url(poll_json)
            error_message = self._find_error_message(poll_json)

            if status_str in failed_statuses:
                print(f"[Shaobkj-SD2.0] 任务失败详情: {self._stringify_payload(poll_json)}")
                if error_message:
                    raise RuntimeError(f"任务失败，{error_message}")
                raise RuntimeError("任务失败")

            if status_str in done_statuses or (progress_val is not None and progress_val >= 99.0):
                if video_url:
                    pbar.update_absolute(90)
                    dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                    return self.download_video(
                        video_url,
                        poll_json,
                        session=session,
                        headers=headers,
                        api_origin=api_origin,
                        timeout_budget=dl_budget,
                    )
                pbar.update_absolute(90)
                dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                return self.download_video(
                    content_url,
                    poll_json,
                    session=session,
                    headers=headers,
                    api_origin=api_origin,
                    timeout_budget=dl_budget,
                )

            if progress_val is not None:
                pbar_val = min(90, 40 + int(float(progress_val) * 0.5))
                pbar.update_absolute(pbar_val)
            else:
                max_attempts = 120 if int(等待时间) == 0 else max(1, int(timeout_val / max(1, poll_interval)))
                pbar_val = min(80, 40 + (attempts * 40 // max_attempts))
                pbar.update_absolute(pbar_val)

    @staticmethod
    def _strip_code_fence(text):
        if not isinstance(text, str):
            return text
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    @classmethod
    def _parse_json_text(cls, text):
        cleaned = cls._strip_code_fence(text)
        if not cleaned:
            return None
        try:
            return json.loads(cleaned)
        except Exception:
            return None

    @classmethod
    def _to_payload(cls, value):
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            parsed = cls._parse_json_text(value)
            if parsed is not None:
                return parsed
            return {"raw": value}
        return {"raw": str(value)}

    @classmethod
    def _to_payload_from_response(cls, response):
        try:
            return cls._to_payload(response.json())
        except Exception:
            try:
                return cls._to_payload(response.text)
            except Exception:
                return {"raw": ""}

    @classmethod
    def _extract_choice_content_payload(cls, payload):
        if not isinstance(payload, dict):
            return None
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0] if isinstance(choices[0], dict) else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return cls._to_payload(content)
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item.get("text"))
            if texts:
                return cls._to_payload("\n".join(texts))
        return None

    @classmethod
    def _walk(cls, value):
        yield value
        if isinstance(value, dict):
            for item in value.values():
                yield from cls._walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from cls._walk(item)
        elif isinstance(value, str):
            parsed = cls._parse_json_text(value)
            if parsed is not None:
                yield from cls._walk(parsed)

    @classmethod
    def _find_task_id(cls, payload):
        if payload is None:
            return None
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue
            candidate = item.get("task_id") or item.get("taskId")
            if candidate is not None:
                return str(candidate)
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue
            if item.get("object") == "chat.completion":
                continue
            candidate = item.get("id")
            if candidate is None:
                continue
            candidate_text = str(candidate).strip()
            if candidate_text and not cls._is_chat_completion_id(candidate_text):
                return candidate_text
        return None

    @staticmethod
    def _is_chat_completion_id(task_id):
        return isinstance(task_id, str) and task_id.startswith("chatcmpl-")

    @classmethod
    def _find_video_url(cls, payload):
        if payload is None:
            return None
        for item in cls._walk(payload):
            if isinstance(item, dict):
                for key in ("video_url", "url", "download_url", "content_url", "video"):
                    value = item.get(key)
                    if isinstance(value, str) and value.startswith(("http://", "https://")):
                        return value
            elif isinstance(item, str):
                match = re.search(r"(https?://[^\s\)\]\"']+)", item)
                if match:
                    return match.group(1).rstrip(".,;!?")
        return None

    @classmethod
    def _find_status(cls, payload):
        if payload is None:
            return ""
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue
            for key in ("status", "task_status", "state"):
                value = item.get(key)
                if value is not None:
                    return str(value).strip().upper()
        return ""

    @classmethod
    def _find_progress(cls, payload):
        if payload is None:
            return None
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue
            for key in ("progress", "percent", "percentage"):
                value = item.get(key)
                if value is None:
                    continue
                try:
                    progress_val = float(str(value).strip().replace("%", ""))
                    if progress_val <= 1.0:
                        progress_val *= 100.0
                    if progress_val < 0:
                        progress_val = 0.0
                    if progress_val > 100:
                        progress_val = 100.0
                    return progress_val
                except Exception:
                    continue
        return None

    @classmethod
    def _find_error_message(cls, payload):
        if payload is None:
            return None
        for item in cls._walk(payload):
            if isinstance(item, dict):
                for key in ("fail_reason", "reason", "message", "error_message", "msg", "detail"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                error_value = item.get("error")
                if isinstance(error_value, str) and error_value.strip():
                    return error_value.strip()
                if isinstance(error_value, dict):
                    nested = cls._find_error_message(error_value)
                    if nested:
                        return nested
        return None

    @staticmethod
    def _stringify_payload(payload):
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)

    def download_video(self, video_url, full_response, session=None, headers=None, api_origin="", timeout_budget=None):
        print(f"[Shaobkj-SD2.0] Downloading video from {video_url}...")
        try:
            dl_headers = auth_headers_for_same_origin(str(video_url), api_origin, headers or {})
            dl_timeout = 60 if timeout_budget is None else max(1, int(timeout_budget))
            if session:
                if dl_headers:
                    v_resp = session.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
                else:
                    v_resp = session.get(video_url, stream=True, verify=False, timeout=dl_timeout)
            else:
                if dl_headers:
                    v_resp = requests.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
                else:
                    v_resp = requests.get(video_url, stream=True, verify=False, timeout=dl_timeout)
            v_resp.raise_for_status()

            filename = f"sd20_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
            output_dir = folder_paths.get_output_directory()
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in v_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            video_obj = InputImpl.VideoFromFile(file_path)
            return (video_obj, self._stringify_payload(full_response))
        except Exception as e:
            print(f"[Shaobkj-SD2.0] 加载视频帧失败: {e}")
            raise
