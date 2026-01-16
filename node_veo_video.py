import os
import json
import requests
import time
import traceback
import random
import io
import torch
import numpy as np
from PIL import Image
from urllib.parse import urlparse
import folder_paths
from .shaobkj_shared import get_config_value
from comfy_api.latest import InputImpl
from comfy.utils import ProgressBar


class Shaobkj_Veo_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型": (["veo_3_1", "veo_3_1-fast"], {"default": "veo_3_1"}),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "任务类型": (["智能模式", "文生视频", "图生视频"], {"default": "智能模式"}),
                "提示词": ("STRING", {"multiline": True, "default": "画面动起来"}),
                "生成时长": (["5", "8", "10", "15"], {"default": "8"}),
                "分辨率": (["9:16", "16:9"], {"default": "9:16"}),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "参考图": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "STRING")
    RETURN_NAMES = ("images", "video", "API响应")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def tensor2pil(self, image):
        t = image
        if isinstance(t, torch.Tensor) and t.dim() == 4:
            t = t[0]
        if isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
        return Image.fromarray(np.clip(255.0 * arr, 0, 255).astype(np.uint8))

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

    def generate_video(self, API密钥, API地址, 模型, 使用系统代理, 任务类型, 提示词, 生成时长, 分辨率, 长边设置, 等待时间, seed, 参考图=None, **kwargs):
        if not API密钥:
            raise ValueError("API Key is required.")

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
            raise RuntimeError(f"API Error {status_code}: {payload}")

        size_map = {
            "9:16": "720x1280",
            "16:9": "1280x720",
        }
        api_size = size_map.get(分辨率, "1280x720")

        base_url = str(API地址).rstrip("/")
        api_origin = urlparse(base_url).netloc
        if base_url.endswith("/v1"):
            api_url = f"{base_url}/video/generations"
        else:
            api_url = f"{base_url}/v1/video/generations"

        payload_data = {
            "model": 模型,
            "group": "default",
            "prompt": 提示词,
            "seconds": str(生成时长),
            "size": api_size,
            "watermark": "false",
            "private": "false",
            "seed": str(seed),
        }

        files = {}
        final_mode = 任务类型
        if 任务类型 == "智能模式":
            final_mode = "图生视频" if 参考图 is not None else "文生视频"

        if final_mode == "图生视频":
            if 参考图 is None:
                raise ValueError("选择'图生视频'模式时，必须连接'参考图'输入。")
            pil_img = self.resize_pil_long_side(self.tensor2pil(参考图), 长边设置)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            files["input_reference"] = ("reference_image.png", buffered.getvalue(), "image/png")

        timeout_val = None if int(等待时间) == 0 else int(等待时间)

        headers = {"Authorization": f"Bearer {API密钥}"}

        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

        session = requests.Session()
        session.trust_env = bool(使用系统代理)
        if not 使用系统代理:
            session.proxies = {}
        proxies = {} if not 使用系统代理 else None

        print(f"[Shaobkj-Veo] Sending request to {api_url}...")

        try:
            resp = session.post(
                api_url,
                headers=headers,
                data=payload_data,
                files=files if files else None,
                verify=False,
                timeout=timeout_val,
                proxies=proxies
            )
        except Exception as e:
            error_msg = f"Connection Failed: {str(e)}\n{traceback.format_exc()}"
            print(f"[Shaobkj-Veo] {error_msg}")
            raise RuntimeError(f"Connection Failed: {str(e)}") from e

        if resp.status_code != 200:
            try:
                err_msg = resp.json()
            except Exception:
                err_msg = resp.text
            
            error_text = f"API Error {resp.status_code}: {err_msg}"
            print(f"[Shaobkj-Veo] {error_text}")
            raise RuntimeError(f"API Error {resp.status_code}: {err_msg}")

        resp_json = resp.json()

        task_id = resp_json.get("id")
        if not task_id:
            if "data" in resp_json and isinstance(resp_json["data"], list) and resp_json["data"] and "url" in resp_json["data"][0]:
                return self.download_video(resp_json["data"][0]["url"], resp_json, session=session, headers=headers, api_origin=api_origin)
            if "url" in resp_json:
                return self.download_video(resp_json["url"], resp_json, session=session, headers=headers, api_origin=api_origin)
            if "data" in resp_json and isinstance(resp_json["data"], dict) and "url" in resp_json["data"]:
                return self.download_video(resp_json["data"]["url"], resp_json, session=session, headers=headers, api_origin=api_origin)
            raise RuntimeError(f"未找到任务ID (id) 或视频链接，API响应: {resp_json}")

        print(f"[Shaobkj-Veo] 任务ID: {task_id}. 开始轮询状态...")

        poll_url = f"{api_url}/{task_id}"
        timeout_val = 86400 if 等待时间 == 0 else 等待时间
        start_time = time.time()
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        def find_video_url(obj):
            if obj is None:
                return None
            if isinstance(obj, str):
                u = obj.strip()
                if u.startswith("http"):
                    return u
                return None
            if isinstance(obj, dict):
                if obj.get("video_url"):
                    return str(obj["video_url"])
                if obj.get("url"):
                    u = str(obj["url"])
                    if u.startswith("http"):
                        return u
                for v in obj.values():
                    res = find_video_url(v)
                    if res:
                        return res
                return None
            if isinstance(obj, list):
                for item in obj:
                    res = find_video_url(item)
                    if res:
                        return res
                return None
            return None

        fail_count = 0
        while True:
            if time.time() - start_time > timeout_val:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")

            time.sleep(5)

            try:
                poll_resp = session.get(
                    poll_url,
                    headers=headers,
                    params={"_t": int(time.time() * 1000)},
                    verify=False,
                    timeout=30,
                    proxies=proxies
                )
                fail_count = 0
            except Exception as e:
                fail_count += 1
                print(f"[Shaobkj-Veo] Polling connection error (attempt {fail_count}): {e}")
                traceback.print_exc()
                if fail_count >= 10:
                    raise RuntimeError(f"Polling failed 10 times consecutively. Last error: {e}")
                time.sleep(2)
                continue

            if poll_resp.status_code != 200:
                try:
                    err_msg = poll_resp.json()
                except Exception:
                    err_msg = poll_resp.text
                raise_if_quota_error(poll_resp.status_code, err_msg)

            poll_json = poll_resp.json()

            status = poll_json.get("status") or poll_json.get("task_status")
            if not status and "data" in poll_json and isinstance(poll_json["data"], dict):
                status = poll_json["data"].get("status")
            status_str = str(status).strip().upper() if status is not None else ""

            progress = "Unknown"
            if "progress" in poll_json:
                progress = poll_json["progress"]
            elif "data" in poll_json and isinstance(poll_json["data"], dict) and "progress" in poll_json["data"]:
                progress = poll_json["data"]["progress"]

            prog_val = None
            try:
                prog_val = float(str(progress).strip().replace("%", ""))
                if prog_val <= 1.0:
                    prog_val = prog_val * 100.0
                if prog_val < 0:
                    prog_val = 0
                if prog_val > 100:
                    prog_val = 100
            except Exception:
                prog_val = None

            video_url = find_video_url(poll_json)
            done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
            if status_str in done_statuses or (prog_val is not None and prog_val >= 99.0):
                if video_url:
                    pbar.update_absolute(100)
                    return self.download_video(video_url, poll_json, session=session, headers=headers, api_origin=api_origin)
                raise RuntimeError(f"生成成功但未找到视频链接: {poll_json}")
            if status_str in ["FAILED", "FAIL", "ERROR"]:
                raise RuntimeError(f"任务失败: {poll_json}")
            if prog_val is not None:
                pbar.update_absolute(min(int(prog_val), 99))

    def download_video(self, video_url, full_response, session=None, headers=None, api_origin=""):
        print(f"[Shaobkj-Veo] Downloading video from {video_url}...")
        file_path = ""
        try:
            send_headers = False
            try:
                send_headers = bool(headers) and bool(api_origin) and urlparse(str(video_url)).netloc == api_origin
            except Exception:
                send_headers = False

            if session:
                if send_headers:
                    v_resp = session.get(video_url, headers=headers, stream=True, verify=False, timeout=60)
                else:
                    v_resp = session.get(video_url, stream=True, verify=False, timeout=60)
            else:
                if send_headers:
                    v_resp = requests.get(video_url, headers=headers, stream=True, verify=False, timeout=60)
                else:
                    v_resp = requests.get(video_url, stream=True, verify=False, timeout=60)
            v_resp.raise_for_status()

            filename = f"veo_{int(time.time())}_{random.randint(1000,9999)}.mp4"
            output_dir = folder_paths.get_output_directory()
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in v_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            import imageio
            reader = imageio.get_reader(file_path, "ffmpeg")
            preview = None
            try:
                preview = reader.get_data(0)
            except Exception:
                preview = None
            if preview is None:
                frame_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            else:
                frame_tensor = torch.from_numpy(np.array(preview)).float().unsqueeze(0) / 255.0

            video_obj = InputImpl.VideoFromFile(file_path)
            return (frame_tensor, video_obj, json.dumps(full_response, ensure_ascii=False))

        except Exception as e:
            print(f"[Shaobkj-Veo] 加载视频帧失败: {e}")
            raise
