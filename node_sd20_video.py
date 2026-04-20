import io
import json
import mimetypes
import os
import random
import re
import time
import traceback
import wave
from urllib.parse import urlparse

import folder_paths
import requests
import torch
from PIL import Image
from comfy.utils import ProgressBar
from comfy_api.latest import InputImpl

from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    pil_to_tensor,
    post_with_retry,
    resize_and_encode_image,
    tensor_to_pil,
)


class Shaobkj_SD20_Video:
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
                    ["doubao-seedance-2-0-260128", "doubao-seedance-2-0-fast-260128"],
                    {"default": "doubao-seedance-2-0-260128", "tooltip": "Seedance 2.0 模型；推荐：doubao-seedance-2-0-260128"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "tooltip": "视频内容描述；推荐：简洁具体"}),
                "生成时长": ("INT", {"default": 5, "min": 4, "max": 15, "step": 1, "tooltip": "请填写视频生成时长，最多15秒；单位：秒"}),
                "画幅比例": (
                    ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "adaptive"],
                    {"default": "16:9", "tooltip": "视频画幅比例；推荐：16:9"},
                ),
                "分辨率": (
                    ["720p", "480p", "native1080p", "1080p", "2k", "4k"],
                    {"default": "720p", "tooltip": "视频分辨率；推荐：720p"},
                ),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0 为无限等待；推荐：0"}),
                "生成音频": ("BOOLEAN", {"default": True, "tooltip": "是否生成音频；推荐：开启"}),
                "返回末帧": ("BOOLEAN", {"default": False, "tooltip": "是否返回末帧图像；推荐：关闭"}),
                "联网搜索": ("BOOLEAN", {"default": False, "tooltip": "是否启用 web_search；推荐：关闭"}),
                "水印": ("BOOLEAN", {"default": False, "tooltip": "是否添加水印；推荐：关闭"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "随机种子；-1 为自动"}),
            },
            "optional": {
                "首帧": ("IMAGE", {"tooltip": "可选首帧图像；推荐：首尾帧模式连接"}),
                "末帧": ("IMAGE", {"tooltip": "可选末帧图像；推荐：首尾帧模式连接"}),
                "参考图1": ("IMAGE", {"tooltip": "参考图 1"}),
                "参考视频1": ("VIDEO", {"tooltip": "参考视频 1"}),
                "参考音频1": ("AUDIO", {"tooltip": "参考音频 1"}),
                "资产包": ("STRING", {"default": "", "multiline": True, "tooltip": "可选资源包 JSON；可填写 asset_id 或 URL"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "任务ID", "API响应", "视频链接", "末帧图像")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_video(
        self,
        API密钥,
        API地址,
        模型,
        使用系统代理,
        提示词,
        生成时长,
        画幅比例,
        分辨率,
        长边设置,
        等待时间,
        生成音频,
        返回末帧,
        联网搜索,
        水印,
        seed,
        首帧=None,
        末帧=None,
        参考图1=None,
        参考视频1=None,
        参考音频1=None,
        资产包="",
        **kwargs,
    ):
        if not API密钥:
            raise ValueError("API Key is required.")

        pbar = ProgressBar(100)
        pbar.update_absolute(0)
        blank_tensor = pil_to_tensor(Image.new("RGB", (1, 1), color="black"))

        base_url = str(API地址).rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        api_origin = urlparse(base_url).netloc
        submit_url = f"{base_url}/seedance/v3/contents/generations/tasks"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API密钥}",
        }

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

        try:
            asset_first_frame, asset_last_frame, asset_ref_images, asset_ref_videos, asset_ref_audios = self._parse_asset_bundle_only(资产包)

            content = [{"type": "text", "text": str(提示词)}]
            frame_count = 0

            first_frame_added = self._append_image_content(
                content,
                首帧,
                int(长边设置),
                role="first_frame",
            )
            if not first_frame_added:
                first_frame_added = self._append_asset_content(content, asset_first_frame, "image_url", "first_frame")
            if first_frame_added:
                frame_count += 1

            last_frame_added = False
            if 末帧 is not None and first_frame_added:
                last_frame_added = self._append_image_content(
                    content,
                    末帧,
                    int(长边设置),
                    role="last_frame",
                )
            elif 末帧 is not None:
                print("[Shaobkj-SD2.0] 检测到末帧，但未提供首帧，已忽略末帧输入。")
            elif first_frame_added:
                last_frame_added = self._append_asset_content(content, asset_last_frame, "image_url", "last_frame")
            if last_frame_added:
                frame_count += 1

            reference_images = []
            legacy_reference_image = kwargs.get("参考图")
            if legacy_reference_image is not None:
                reference_images.append(legacy_reference_image)
            reference_images.append(参考图1)
            reference_images.extend(self._collect_sorted_kwargs(kwargs, "参考图"))
            reference_images = reference_images[:9]

            ref_image_count = 0
            for image_tensor in reference_images:
                if self._append_image_content(content, image_tensor, int(长边设置), role="reference_image"):
                    ref_image_count += 1
            for asset_ref in self._split_asset_refs(asset_ref_images):
                if self._append_asset_content(content, asset_ref, "image_url", "reference_image"):
                    ref_image_count += 1

            pbar.update_absolute(18)

            reference_videos = [参考视频1]
            reference_videos.extend(self._collect_sorted_kwargs(kwargs, "参考视频"))
            reference_videos.extend(self._collect_sorted_kwargs(kwargs, "video_"))
            reference_videos = reference_videos[:3]
            ref_video_count = 0
            for video_input in reference_videos:
                video_url = self._upload_video_get_url(session, base_url, API密钥, video_input, proxies=proxies)
                if video_url:
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": video_url},
                            "role": "reference_video",
                        }
                    )
                    ref_video_count += 1
            for asset_ref in self._split_asset_refs(asset_ref_videos):
                if self._append_asset_content(content, asset_ref, "video_url", "reference_video"):
                    ref_video_count += 1

            pbar.update_absolute(28)

            reference_audios = [参考音频1]
            reference_audios.extend(self._collect_sorted_kwargs(kwargs, "参考音频"))
            reference_audios = reference_audios[:3]
            ref_audio_count = 0
            for audio_input in reference_audios:
                audio_url = self._upload_audio_get_url(session, base_url, API密钥, audio_input, proxies=proxies)
                if audio_url:
                    content.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_url},
                            "role": "reference_audio",
                        }
                    )
                    ref_audio_count += 1
            for asset_ref in self._split_asset_refs(asset_ref_audios):
                if self._append_asset_content(content, asset_ref, "audio_url", "reference_audio"):
                    ref_audio_count += 1

            pbar.update_absolute(35)

            payload = {
                "model": str(模型).strip(),
                "content": content,
                "duration": int(生成时长),
                "ratio": str(画幅比例).strip(),
                "resolution": str(分辨率).strip(),
                "generate_audio": bool(生成音频),
                "return_last_frame": bool(返回末帧),
                "watermark": bool(水印),
            }
            if 联网搜索:
                payload["tools"] = [{"type": "web_search"}]
            if int(seed) >= 0:
                payload["seed"] = int(seed)

            print(
                f"[Shaobkj-SD2.0] 提交任务 model={模型}, duration={生成时长}s, ratio={画幅比例}, "
                f"resolution={分辨率}, frames={frame_count}, ref_images={ref_image_count}, "
                f"ref_videos={ref_video_count}, ref_audios={ref_audio_count}"
            )

            resp = post_with_retry(
                session,
                submit_url,
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
            err_msg = self._response_payload(resp)
            print(f"[Shaobkj-SD2.0] API Error {resp.status_code}: {self._stringify_payload(err_msg)}")
            raise RuntimeError(f"API Error {resp.status_code}: {self._stringify_payload(err_msg)}")

        submit_payload = self._response_payload(resp)
        task_id = str(submit_payload.get("id") or submit_payload.get("task_id") or "").strip()
        direct_video_url, direct_last_frame_url = self._extract_media_urls(submit_payload)
        submit_status = self._normalize_status(submit_payload.get("status") or submit_payload.get("state"))

        if direct_video_url and submit_status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"}:
            pbar.update_absolute(92)
            video_obj = self._download_video(
                direct_video_url,
                session=session,
                headers=headers,
                api_origin=api_origin,
                timeout_budget=None if int(等待时间) == 0 else int(等待时间),
            )
            last_frame_tensor = blank_tensor
            if bool(返回末帧) and direct_last_frame_url:
                downloaded_last_frame = self._download_image(
                    direct_last_frame_url,
                    session=session,
                    headers=headers,
                    api_origin=api_origin,
                    timeout_budget=None if int(等待时间) == 0 else int(等待时间),
                )
                if downloaded_last_frame is not None:
                    last_frame_tensor = downloaded_last_frame
            return (
                video_obj,
                task_id,
                self._stringify_payload(submit_payload),
                direct_video_url,
                last_frame_tensor,
            )

        if not task_id:
            print(f"[Shaobkj-SD2.0] 未找到任务ID详情: {self._stringify_payload(submit_payload)}")
            raise RuntimeError("未找到任务ID (id/task_id)")

        status_url = f"{base_url}/seedance/v3/contents/generations/tasks/{task_id}"
        timeout_val = 86400 if int(等待时间) == 0 else int(等待时间)
        start_time = time.time()
        poll_interval = 5
        fail_count = 0
        pbar.update_absolute(40)

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_val - elapsed
            if remaining <= 0:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")

            time.sleep(min(poll_interval, max(0.0, remaining)))
            poll_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))

            try:
                poll_resp = session.get(
                    status_url,
                    headers={"Authorization": f"Bearer {API密钥}"},
                    params={"_t": int(time.time() * 1000)},
                    verify=False,
                    timeout=poll_timeout,
                    proxies=proxies,
                )
            except Exception as e:
                fail_count += 1
                print(f"[Shaobkj-SD2.0] 轮询失败第 {fail_count} 次: {e}")
                if fail_count >= 10:
                    raise RuntimeError(f"轮询连续失败 10 次: {e}") from e
                continue

            fail_count = 0
            if poll_resp.status_code != 200:
                err_msg = self._response_payload(poll_resp)
                raise RuntimeError(f"API Error {poll_resp.status_code}: {self._stringify_payload(err_msg)}")

            poll_payload = self._response_payload(poll_resp)
            status = self._find_status(poll_payload)
            progress_val = self._find_progress(poll_payload)
            video_url, last_frame_url = self._extract_media_urls(poll_payload)
            error_message = self._find_error_message(poll_payload)

            if status in {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}:
                print(f"[Shaobkj-SD2.0] 任务失败详情: {self._stringify_payload(poll_payload)}")
                if error_message:
                    raise RuntimeError(f"任务失败，{error_message}")
                raise RuntimeError("任务失败")

            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "DONE", "FINISHED"} and video_url:
                pbar.update_absolute(92)
                dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                video_obj = self._download_video(
                    video_url,
                    session=session,
                    headers=headers,
                    api_origin=api_origin,
                    timeout_budget=dl_budget,
                )
                last_frame_tensor = blank_tensor
                if bool(返回末帧) and last_frame_url:
                    downloaded_last_frame = self._download_image(
                        last_frame_url,
                        session=session,
                        headers=headers,
                        api_origin=api_origin,
                        timeout_budget=dl_budget,
                    )
                    if downloaded_last_frame is not None:
                        last_frame_tensor = downloaded_last_frame
                pbar.update_absolute(100)
                return (
                    video_obj,
                    task_id,
                    self._stringify_payload(poll_payload),
                    video_url,
                    last_frame_tensor,
                )

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
    def _response_payload(response):
        try:
            data = response.json()
            return data if isinstance(data, dict) else {"data": data}
        except Exception:
            return {"raw": response.text}

    @staticmethod
    def _split_asset_refs(value):
        if not value or not str(value).strip():
            return []
        return [item.strip() for item in re.split(r"[\n,]+", str(value)) if item.strip()]

    @classmethod
    def _parse_asset_bundle_only(cls, bundle_json):
        if not bundle_json or not str(bundle_json).strip():
            return "", "", "", "", ""
        try:
            bundle = json.loads(str(bundle_json).strip())
        except Exception as e:
            print(f"[Shaobkj-SD2.0] 资产包 JSON 解析失败: {e}")
            return "", "", "", "", ""

        def join_list(key):
            value = bundle.get(key)
            if isinstance(value, list):
                return ",".join(str(item).strip() for item in value if str(item).strip())
            if isinstance(value, str):
                return value.strip()
            return ""

        return (
            str(bundle.get("first_frame") or "").strip(),
            str(bundle.get("last_frame") or "").strip(),
            join_list("ref_images"),
            join_list("videos"),
            join_list("audios"),
        )

    @staticmethod
    def _asset_ref_to_url(value):
        text = str(value or "").strip()
        if not text:
            return None
        lower_text = text.lower()
        if lower_text.startswith("http://") or lower_text.startswith("https://") or lower_text.startswith("asset://"):
            return text
        return f"asset://{text}"

    @classmethod
    def _append_asset_content(cls, content, asset_ref, content_type, role):
        resolved = cls._asset_ref_to_url(asset_ref)
        if not resolved:
            return False
        content.append(
            {
                "type": content_type,
                content_type: {"url": resolved},
                "role": role,
            }
        )
        return True

    @staticmethod
    def _collect_sorted_kwargs(kwargs, prefix):
        items = []
        for key, value in kwargs.items():
            if value is None or not str(key).startswith(prefix):
                continue
            suffix = str(key)[len(prefix):]
            if suffix == "":
                continue
            try:
                index = int(suffix)
            except Exception:
                index = 999999
            items.append((index, key, value))
        items.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in items]

    @staticmethod
    def _image_tensor_to_data_url(image_tensor, long_side):
        if image_tensor is None:
            return None
        image_b64, _ = resize_and_encode_image(tensor_to_pil(image_tensor), int(long_side))
        if not image_b64:
            return None
        return f"data:image/jpeg;base64,{image_b64}"

    @classmethod
    def _append_image_content(cls, content, image_tensor, long_side, role):
        data_url = cls._image_tensor_to_data_url(image_tensor, long_side)
        if not data_url:
            return False
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
                "role": role,
            }
        )
        return True

    @staticmethod
    def _file_input_to_bytes(media_input, default_ext, label):
        if media_input is None:
            return None, None

        get_stream = getattr(media_input, "get_stream_source", None)
        if callable(get_stream):
            try:
                source = media_input.get_stream_source()
                if isinstance(source, str):
                    source = source.strip()
                    if source and os.path.isfile(source):
                        with open(source, "rb") as f:
                            return f.read(), os.path.basename(source)
                elif isinstance(source, io.BytesIO):
                    source.seek(0)
                    data = source.read()
                    if data:
                        return data, f"reference_{label}_{abs(hash(data)) % 10**10}{default_ext}"
                elif hasattr(source, "read"):
                    if hasattr(source, "seek"):
                        source.seek(0)
                    data = source.read()
                    if data:
                        return data, f"reference_{label}_{abs(hash(data)) % 10**10}{default_ext}"
            except Exception as e:
                print(f"[Shaobkj-SD2.0] 读取 {label} 流失败: {e}")

        if isinstance(media_input, str):
            path = media_input.strip()
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read(), os.path.basename(path)

        if isinstance(media_input, dict):
            path = (
                media_input.get("path")
                or media_input.get("file")
                or media_input.get("file_path")
                or media_input.get("filename")
                or ""
            )
            path = str(path).strip()
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read(), os.path.basename(path)

        for attr in ("path", "file_path"):
            path = getattr(media_input, attr, None)
            if isinstance(path, str):
                path = path.strip()
                if path and os.path.isfile(path):
                    with open(path, "rb") as f:
                        return f.read(), os.path.basename(path)

        return None, None

    @staticmethod
    def _waveform_to_wav_bytes(waveform, sample_rate):
        waveform = waveform.detach().cpu().float()
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform [C, T], got shape {tuple(waveform.shape)}")
        channels, _ = waveform.shape
        waveform = waveform.clamp(-1.0, 1.0)
        pcm = (waveform.numpy() * 32767.0).astype("int16")
        interleaved = pcm[0] if channels == 1 else pcm.transpose(1, 0).reshape(-1)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(int(channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(interleaved.tobytes())
        return buffer.getvalue()

    @classmethod
    def _audio_input_to_bytes(cls, audio_input):
        file_content, filename = cls._file_input_to_bytes(audio_input, ".wav", "audio")
        if file_content:
            return file_content, filename

        if isinstance(audio_input, dict) and audio_input.get("waveform") is not None:
            waveform = audio_input.get("waveform")
            if torch.is_tensor(waveform):
                sample_rate = int(audio_input.get("sample_rate", 44100))
                return cls._waveform_to_wav_bytes(waveform, sample_rate), "reference_audio.wav"

        return None, None

    @staticmethod
    def _upload_file(session, base_url, api_key, file_content, filename, content_type, proxies=None):
        if not file_content:
            return None
        upload_headers = {"Authorization": f"Bearer {api_key}"}
        response = session.post(
            f"{base_url}/v1/files",
            headers=upload_headers,
            files={"file": (filename, file_content, content_type)},
            timeout=120,
            proxies=proxies,
            verify=False,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("url")

    @classmethod
    def _upload_video_get_url(cls, session, base_url, api_key, video_input, proxies=None):
        file_content, filename = cls._file_input_to_bytes(video_input, ".mp4", "video")
        if not file_content:
            return None
        if not filename:
            filename = f"reference_video_{abs(hash(file_content)) % 10**10}.mp4"
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "video/mp4"
        try:
            return cls._upload_file(session, base_url, api_key, file_content, filename, mime_type, proxies=proxies)
        except Exception as e:
            print(f"[Shaobkj-SD2.0] 上传参考视频失败: {e}")
            return None

    @classmethod
    def _upload_audio_get_url(cls, session, base_url, api_key, audio_input, proxies=None):
        file_content, filename = cls._audio_input_to_bytes(audio_input)
        if not file_content:
            return None
        if not filename:
            filename = f"reference_audio_{abs(hash(file_content)) % 10**10}.wav"
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "audio/wav"
        try:
            return cls._upload_file(session, base_url, api_key, file_content, filename, mime_type, proxies=proxies)
        except Exception as e:
            print(f"[Shaobkj-SD2.0] 上传参考音频失败: {e}")
            return None

    @staticmethod
    def _normalize_status(status):
        text = str(status or "").strip().upper()
        if text == "SUCCESS":
            return "SUCCEEDED"
        if text in {"FAIL", "FAILURE"}:
            return "FAILED"
        return text

    @classmethod
    def _walk(cls, value):
        yield value
        if isinstance(value, dict):
            for item in value.values():
                yield from cls._walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from cls._walk(item)

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
                    return cls._normalize_status(value)
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
                    return max(0.0, min(100.0, progress_val))
                except Exception:
                    continue
        return None

    @classmethod
    def _find_error_message(cls, payload):
        if payload is None:
            return None
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue
            for key in ("fail_reason", "failReason", "reason", "message", "error_message", "msg", "detail"):
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

    @classmethod
    def _extract_media_urls(cls, payload):
        video_url = None
        last_frame_url = None
        for item in cls._walk(payload):
            if not isinstance(item, dict):
                continue

            root_content = item.get("content")
            if isinstance(root_content, dict):
                if not video_url:
                    video_url = root_content.get("video_url") or root_content.get("videoUrl") or video_url
                if not last_frame_url:
                    last_frame_url = (
                        root_content.get("last_frame_url")
                        or root_content.get("lastFrameUrl")
                        or root_content.get("last_frame_image_url")
                        or last_frame_url
                    )

            if not video_url:
                video_url = item.get("video_url") or item.get("videoUrl") or video_url
            if not last_frame_url:
                last_frame_url = (
                    item.get("last_frame_url")
                    or item.get("lastFrameUrl")
                    or item.get("last_frame_image_url")
                    or last_frame_url
                )

            results = item.get("results")
            if not video_url and isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    result_url = result.get("url")
                    result_type = str(result.get("outputType") or "").lower()
                    if isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                        if result_type in {"mp4", "video"} or result_url.lower().endswith(".mp4"):
                            video_url = result_url
                            break
                        if not video_url:
                            video_url = result_url

            content_list = item.get("content")
            if isinstance(content_list, list):
                for entry in content_list:
                    if not isinstance(entry, dict):
                        continue
                    entry_type = entry.get("type")
                    entry_role = entry.get("role")
                    if entry_type == "video_url" and not video_url:
                        entry_video = entry.get("video_url")
                        if isinstance(entry_video, dict):
                            video_url = entry_video.get("url") or video_url
                        elif isinstance(entry_video, str):
                            video_url = entry_video
                    if entry_type == "image_url" and entry_role == "last_frame" and not last_frame_url:
                        entry_image = entry.get("image_url")
                        if isinstance(entry_image, dict):
                            last_frame_url = entry_image.get("url") or last_frame_url
                        elif isinstance(entry_image, str):
                            last_frame_url = entry_image

            if not last_frame_url:
                last_frame = item.get("last_frame") or item.get("lastFrame")
                if isinstance(last_frame, dict):
                    last_frame_url = last_frame.get("url") or last_frame_url
                elif isinstance(last_frame, str) and last_frame.startswith(("http://", "https://")):
                    last_frame_url = last_frame

        return video_url, last_frame_url

    def _download_video(self, video_url, session=None, headers=None, api_origin="", timeout_budget=None):
        print(f"[Shaobkj-SD2.0] Downloading video from {video_url}...")
        dl_headers = auth_headers_for_same_origin(str(video_url), api_origin, headers or {})
        dl_timeout = 60 if timeout_budget is None else max(1, int(timeout_budget))
        if session:
            if dl_headers:
                response = session.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            else:
                response = session.get(video_url, stream=True, verify=False, timeout=dl_timeout)
        else:
            if dl_headers:
                response = requests.get(video_url, headers=dl_headers, stream=True, verify=False, timeout=dl_timeout)
            else:
                response = requests.get(video_url, stream=True, verify=False, timeout=dl_timeout)
        response.raise_for_status()

        filename = f"sd20_{int(time.time())}_{random.randint(1000, 9999)}.mp4"
        output_dir = folder_paths.get_output_directory()
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return InputImpl.VideoFromFile(file_path)

    def _download_image(self, image_url, session=None, headers=None, api_origin="", timeout_budget=None):
        dl_headers = auth_headers_for_same_origin(str(image_url), api_origin, headers or {})
        dl_timeout = 30 if timeout_budget is None else max(1, min(30, int(timeout_budget)))
        if session:
            if dl_headers:
                response = session.get(image_url, headers=dl_headers, verify=False, timeout=dl_timeout)
            else:
                response = session.get(image_url, verify=False, timeout=dl_timeout)
        else:
            if dl_headers:
                response = requests.get(image_url, headers=dl_headers, verify=False, timeout=dl_timeout)
            else:
                response = requests.get(image_url, verify=False, timeout=dl_timeout)
        response.raise_for_status()
        return pil_to_tensor(Image.open(io.BytesIO(response.content)).convert("RGB"))
