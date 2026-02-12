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
from .shaobkj_shared import (
    auth_headers_for_same_origin,
    build_submit_timeout,
    create_requests_session,
    disable_insecure_request_warnings,
    extract_task_id_and_video_url,
    get_config_value,
    post_with_retry,
    resize_pil_long_side,
    tensor_to_pil,
)
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
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型": (["veo_3_1", "veo_3_1-fast"], {"default": "veo_3_1", "tooltip": "视频模型选择；推荐：veo_3_1"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "任务类型": (["智能模式", "文生视频", "图生视频"], {"default": "智能模式", "tooltip": "生成模式；推荐：智能模式"}),
                "提示词": ("STRING", {"multiline": True, "default": "画面动起来", "tooltip": "视频内容描述；推荐：简洁具体"}),
                "生成时长": (["5", "8", "10", "15"], {"default": "8", "tooltip": "视频时长(秒)；推荐：8"}),
                "分辨率": (["9:16", "16:9"], {"default": "9:16", "tooltip": "视频画幅比例；推荐：9:16"}),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待；推荐：0"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False, "tooltip": "API 申请入口；推荐：默认地址"}),
            },
            "optional": {
                "参考图": ("IMAGE", {"tooltip": "图生视频参考图；推荐：图生视频时必填"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "API响应")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_video(self, API密钥, API地址, 模型, 使用系统代理, 任务类型, 提示词, 生成时长, 分辨率, 长边设置, 等待时间, seed, 参考图=None, **kwargs):
        pbar = ProgressBar(100)
        pbar.update_absolute(0)

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
        root_base = base_url[:-3] if base_url.endswith("/v1") else base_url
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
            # ComfyUI 的 seed 是 INT 类型，保持 INT 类型传递给 API
            "seed": seed,
        }

        files = {}
        final_mode = 任务类型
        if 任务类型 == "智能模式":
            final_mode = "图生视频" if 参考图 is not None else "文生视频"

        if final_mode == "图生视频":
            if 参考图 is None:
                raise ValueError("选择'图生视频'模式时，必须连接'参考图'输入。")
            pil_img = resize_pil_long_side(tensor_to_pil(参考图), 长边设置)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            files["input_reference"] = ("reference_image.png", buffered.getvalue(), "image/png")

        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        submit_timeout = build_submit_timeout(int(等待时间))

        headers = {"Authorization": f"Bearer {API密钥}"}

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))

        try:
            pbar.update_absolute(10)
            resp = post_with_retry(
                session,
                api_url,
                headers=headers,
                timeout=submit_timeout,
                proxies=proxies,
                verify=False,
                data=payload_data,
                files=files if files else None,
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

        pbar.update_absolute(30)
        task_id, direct_video_url, parsed_json, raw_text = extract_task_id_and_video_url(resp)
        resp_json = parsed_json if parsed_json is not None else {"raw": raw_text}

        dl_budget = None if int(等待时间) == 0 else max(1, int(等待时间))
        if direct_video_url:
            pbar.update_absolute(90)
            return self.download_video(direct_video_url, resp_json, session=session, headers=headers, api_origin=api_origin, timeout_budget=dl_budget)

        if not task_id:
            pbar.update_absolute(100)
            print(f"[Shaobkj-Veo] 未找到任务ID或视频链接详情: {json.dumps(resp_json, ensure_ascii=False)}")
            raise RuntimeError("未找到任务ID (id/task_id) 或视频链接")

        poll_url = f"{api_url}/{task_id}"
        alt_poll_url = f"{root_base}/v1/videos/{task_id}"
        alt_poll_body = {"model": 模型}
        alt_content_url = f"{root_base}/v1/videos/{task_id}/content"
        timeout_val = 86400 if 等待时间 == 0 else 等待时间
        start_time = time.time()
        pbar.update_absolute(40)
        attempts = 0
        poll_interval = 5

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
            elapsed = time.time() - start_time
            remaining = timeout_val - elapsed
            if remaining <= 0:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")

            time.sleep(min(poll_interval, max(0.0, remaining)))

            try:
                poll_req_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))
                poll_resp = session.get(
                    poll_url,
                    headers=headers,
                    params={"_t": int(time.time() * 1000)},
                    verify=False,
                    timeout=poll_req_timeout,
                    proxies=proxies
                )
                attempts += 1
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
                if poll_resp.status_code in (404, 405):
                    try:
                        alt_resp = session.get(
                            alt_poll_url,
                            headers=headers,
                            json=alt_poll_body,
                            verify=False,
                            timeout=poll_req_timeout,
                            proxies=proxies,
                        )
                        if alt_resp.status_code == 200:
                            poll_resp = alt_resp
                    except Exception:
                        pass
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
            failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}
            if status_str in failed_statuses:
                fail_reason = poll_json.get("fail_reason")
                if not fail_reason and isinstance(poll_json.get("data"), dict):
                    fail_reason = poll_json["data"].get("fail_reason") or poll_json["data"].get("reason")
                inner_err = None
                if isinstance(poll_json.get("data"), dict) and isinstance(poll_json["data"].get("data"), dict):
                    inner_err = poll_json["data"]["data"].get("error")
                if isinstance(inner_err, dict):
                    inner_err = inner_err.get("message") or inner_err.get("code")
                error_message = None
                if isinstance(poll_json.get("error"), dict):
                    error_message = poll_json["error"].get("message") or poll_json["error"].get("code")
                elif isinstance(poll_json.get("data"), dict) and isinstance(poll_json["data"].get("error"), dict):
                    error_message = poll_json["data"]["error"].get("message") or poll_json["data"]["error"].get("code")
                if isinstance(error_message, dict):
                    error_message = error_message.get("message") or error_message.get("code") or json.dumps(error_message, ensure_ascii=False)
                elif error_message is not None:
                    error_message = str(error_message)
                if fail_reason:
                    print(f"[Shaobkj-Veo] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
                    if error_message:
                        raise RuntimeError(f"任务失败，{error_message}")
                    raise RuntimeError(f"任务失败({fail_reason})")
                if inner_err:
                    print(f"[Shaobkj-Veo] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
                    if error_message:
                        raise RuntimeError(f"任务失败，{error_message}")
                    raise RuntimeError(f"任务失败({inner_err})")
                print(f"[Shaobkj-Veo] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
                if error_message:
                    raise RuntimeError(f"任务失败，{error_message}")
                raise RuntimeError("任务失败")

            if status_str in done_statuses or (prog_val is not None and prog_val >= 99.0):
                if not video_url and isinstance(alt_content_url, str) and alt_content_url.startswith("http"):
                    try:
                        pbar.update_absolute(90)
                        dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                        return self.download_video(alt_content_url, poll_json, session=session, headers=headers, api_origin=api_origin, timeout_budget=dl_budget)
                    except Exception:
                        pass
                if video_url:
                    pbar.update_absolute(90)
                    dl_budget = None if int(等待时间) == 0 else max(1, int(timeout_val - (time.time() - start_time)))
                    return self.download_video(video_url, poll_json, session=session, headers=headers, api_origin=api_origin, timeout_budget=dl_budget)
                print(f"[Shaobkj-Veo] 生成成功但未找到视频链接详情: {json.dumps(poll_json, ensure_ascii=False)}")
                raise RuntimeError("生成成功但未找到视频链接")
            if prog_val is not None:
                pbar_val = min(90, 40 + int(float(prog_val) * 0.5))
                pbar.update_absolute(pbar_val)
            else:
                max_attempts = 120 if int(等待时间) == 0 else max(1, int(timeout_val / max(1, poll_interval)))
                pbar_val = min(80, 40 + (attempts * 40 // max_attempts))
                pbar.update_absolute(pbar_val)

    def download_video(self, video_url, full_response, session=None, headers=None, api_origin="", timeout_budget=None):
        print(f"[Shaobkj-Veo] Downloading video from {video_url}...")
        file_path = ""
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

            filename = f"veo_{int(time.time())}_{random.randint(1000,9999)}.mp4"
            output_dir = folder_paths.get_output_directory()
            file_path = os.path.join(output_dir, filename)

            with open(file_path, "wb") as f:
                for chunk in v_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            video_obj = InputImpl.VideoFromFile(file_path)
            return (video_obj, json.dumps(full_response, ensure_ascii=False))

        except Exception as e:
            print(f"[Shaobkj-Veo] 加载视频帧失败: {e}")
            raise
