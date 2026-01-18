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

class Shaobkj_Sora_Video:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型": (["sora-2", "sora-2-pro"], {"default": "sora-2"}),
                "使用系统代理": ("BOOLEAN", {"default": False}),
                "任务类型": (["智能模式", "文生视频", "图生视频", "角色视频"], {"default": "智能模式"}),
                "提示词": ("STRING", {"multiline": True, "default": "画面动起来"}),
                "生成时长": (["10", "15", "25"], {"default": "10"}),
                "分辨率": (["9:16", "16:9"], {"default": "9:16"}),
                "长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "轮询等待时间(秒)，0为无限等待"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
            "optional": {
                "参考图": ("IMAGE",),
                "角色视频URL": ("STRING", {"multiline": False, "default": ""}),
                "角色时间戳": ("STRING", {"multiline": False, "default": "1,3"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "API响应")
    FUNCTION = "generate_video"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_video(self, API密钥, API地址, 模型, 使用系统代理, 任务类型, 提示词, 生成时长, 分辨率, 长边设置, 等待时间, seed, 参考图=None, 角色视频URL="", 角色时间戳="", **kwargs):
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

        # 自动模型切换逻辑：如果选择25秒且使用sora-2，自动切换为sora-2-pro
        if str(生成时长) == "25" and 模型 == "sora-2":
             print("[Shaobkj-Sora] 警告：sora-2 模型可能不支持25秒时长，建议使用 sora-2-pro。")
             # 这里不强制切换，因为用户可能知道API支持，只做日志提示
        
        # 任务类型检查
        if 任务类型 == "图生视频" and 参考图 is None:
            raise ValueError("选择'图生视频'模式时，必须连接'参考图'输入。")
        
        if 任务类型 == "角色视频" and (not 角色视频URL or not 角色视频URL.strip()):
             # 角色视频通常也需要参考图作为人物基准，或者只需要URL？根据文档 character_url 是可选的，但 input_reference 是必须的
             # 假设角色视频模式下，角色URL是必须的，参考图也是必须的（作为首帧或人物参考）
             if 参考图 is None:
                 print("[Shaobkj-Sora] 警告：'角色视频'模式通常需要'参考图'作为人物基准。")
             if not 角色视频URL.strip():
                 print("[Shaobkj-Sora] 警告：'角色视频'模式建议提供'角色视频URL'。")

        # 分辨率映射
        size_map = {
            "9:16": "720x1280",
            "16:9": "1280x720"
        }
        api_size = size_map.get(分辨率, "720x1280")

        # 构造基础 URL
        base_url = str(API地址).rstrip("/")
        root_base = base_url[:-3] if base_url.endswith("/v1") else base_url
        api_origin = urlparse(base_url).netloc
        if base_url.endswith("/v1"):
            api_url = f"{base_url}/video/generations"
        else:
            api_url = f"{base_url}/v1/video/generations"

        # 构造请求参数
        # 文档: https://zhzwx2axs4.apifox.cn/401540802e0
        payload_data = {
            "model": 模型,
            "prompt": 提示词,
            "seconds": str(生成时长),
            "size": api_size,
            "watermark": "false", # 默认无水印
            "private": "false",   # 默认公开
            # ComfyUI 的 seed 是 INT 类型，无需手动转 str，requests/json 会自动处理
            # 保持 INT 类型能确保与 ComfyUI 的逻辑一致，且大多数 JSON API 都接受数字类型的 seed
            "seed": seed,
        }

        # 可选参数
        if 角色视频URL and 角色视频URL.strip():
            payload_data["character_url"] = 角色视频URL.strip()
        
        if 角色时间戳 and 角色时间戳.strip():
            payload_data["character_timestamps"] = 角色时间戳.strip()

        files = {}
        
        # 智能模式判断逻辑
        final_mode = 任务类型
        if 任务类型 == "智能模式":
            has_ref_img = 参考图 is not None
            has_char_url = 角色视频URL and 角色视频URL.strip()
            
            if has_ref_img and has_char_url:
                final_mode = "角色视频"
                print(f"[Shaobkj-Sora] 智能模式: 检测到参考图和角色URL -> 切换为【角色视频】")
            elif has_ref_img:
                final_mode = "图生视频"
                print(f"[Shaobkj-Sora] 智能模式: 仅检测到参考图 -> 切换为【图生视频】")
            else:
                final_mode = "文生视频"
                print(f"[Shaobkj-Sora] 智能模式: 未检测到参考图 -> 切换为【文生视频】")

        # 根据最终模式处理 input_reference
        if final_mode == "文生视频":
            print(f"[Shaobkj-Sora] 执行模式: 文生视频。忽略参考图。")
            # 即使有参考图也不发送，确保是纯文生视频
            pass
        elif (final_mode == "图生视频" or final_mode == "角色视频") and 参考图 is not None:
            print(f"[Shaobkj-Sora] 执行模式: {final_mode}。启用图生视频/角色模式。")
            pil_img = resize_pil_long_side(tensor_to_pil(参考图), 长边设置)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            
            # 严格遵循文档: input_reference
            files["input_reference"] = ("reference_image.png", img_bytes, "image/png")
        elif 参考图 is not None:
            # 兼容性 Fallback (理论上不应到达这里，除非强制选了图生视频但没给图，这在上面已校验)
            print(f"[Shaobkj-Sora] 执行模式: {final_mode} (Fallback)。")
            pil_img = resize_pil_long_side(tensor_to_pil(参考图), 长边设置)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            files["input_reference"] = ("reference_image.png", img_bytes, "image/png")
        else:
             if final_mode != "文生视频":
                 print(f"[Shaobkj-Sora] 警告: 模式为 {final_mode} 但未提供参考图，API 可能会报错。")
             print(f"[Shaobkj-Sora] 执行模式: 文生视频 (无参考图)。")

        headers = {
            "Authorization": f"Bearer {API密钥}"
        }

        timeout_val = None if int(等待时间) == 0 else int(等待时间)
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

        print(f"[Shaobkj-Sora] Sending request to {api_url}...")
        print(f"[Shaobkj-Sora] Payload: {payload_data}")
        
        try:
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
            print(f"[Shaobkj-Sora] {error_msg}")
            raise RuntimeError(f"Connection Failed: {str(e)}") from e

        if resp.status_code != 200:
            try:
                err_msg = resp.json()
            except:
                err_msg = resp.text
            
            error_text = f"API Error {resp.status_code}: {err_msg}"
            print(f"[Shaobkj-Sora] {error_text}")
            raise RuntimeError(f"API Error {resp.status_code}: {err_msg}")

        task_id, direct_video_url, parsed_json, raw_text = extract_task_id_and_video_url(resp)
        resp_json = parsed_json if parsed_json is not None else {"raw": raw_text}
        try:
            print(f"[Shaobkj-Sora] Response: {json.dumps(resp_json, ensure_ascii=False)}")
        except Exception:
            print(f"[Shaobkj-Sora] Response: {str(raw_text)[:500] if raw_text else ''}")

        dl_budget = None if int(等待时间) == 0 else max(1, int(等待时间))
        if direct_video_url:
            return self.download_video(direct_video_url, resp_json, session=session, headers=headers, api_origin=api_origin, timeout_budget=dl_budget)

        if not task_id:
            raise RuntimeError(f"未找到任务ID (id/task_id) 或视频链接，API响应: {resp_json}")

        print(f"[Shaobkj-Sora] 任务ID: {task_id}. 开始轮询状态...")
        
        poll_url = f"{api_url}/{task_id}" 
        alt_poll_url = f"{root_base}/v1/videos/{task_id}"
        alt_poll_body = {"model": 模型}
        alt_content_url = f"{root_base}/v1/videos/{task_id}/content"
        timeout_val = 86400 if 等待时间 == 0 else 等待时间
        start_time = time.time()
        pbar = ProgressBar(100)
        pbar.update_absolute(0)
        poll_interval = 5
        attempts = 0
        last_log_key = None
        
        fail_count = 0
        while True:
            elapsed = time.time() - start_time
            remaining = timeout_val - elapsed
            if remaining <= 0:
                raise RuntimeError(f"视频生成超时 ({timeout_val}秒)")
            
            time.sleep(min(poll_interval, max(0.0, remaining)))
            
            poll_req_timeout = 30 if int(等待时间) == 0 else max(1, min(30, int(remaining)))
            t_params = {"_t": int(time.time() * 1000)}
            poll_resp = None
            last_exc = None
            try:
                poll_resp = session.get(
                    alt_poll_url,
                    headers=headers,
                    json=alt_poll_body,
                    params=t_params,
                    verify=False,
                    timeout=poll_req_timeout,
                    proxies=proxies,
                )
            except Exception as e:
                last_exc = e

            if poll_resp is None or poll_resp.status_code != 200:
                try:
                    poll_resp2 = session.get(
                        poll_url,
                        headers=headers,
                        params=t_params,
                        verify=False,
                        timeout=poll_req_timeout,
                        proxies=proxies,
                    )
                    if poll_resp2.status_code == 200:
                        poll_resp = poll_resp2
                except Exception as e:
                    if poll_resp is None:
                        last_exc = e if last_exc is None else last_exc

            attempts += 1
            if poll_resp is None:
                fail_count += 1
                print(f"[Shaobkj-Sora] Poll Connection Error (attempt {fail_count}): {last_exc}")
                traceback.print_exc()
                if fail_count >= 10:
                    raise RuntimeError(f"Polling failed 10 times consecutively. Last error: {last_exc}")
                time.sleep(2)
                continue

            fail_count = 0
            if poll_resp.status_code != 200:
                try:
                    err_msg = poll_resp.json()
                except Exception:
                    err_msg = poll_resp.text
                print(f"[Shaobkj-Sora] Poll Error {poll_resp.status_code}: {err_msg}")
                continue
            
            poll_json = poll_resp.json()
            # 调试：打印完整的轮询响应 -> 改为仅打印关键状态
            # print(f"[Shaobkj-Sora] Poll Response: {json.dumps(poll_json, ensure_ascii=False)}")
            
            status = poll_json.get("status")
            if not status:
                status = poll_json.get("task_status")
            if not status and "data" in poll_json and isinstance(poll_json["data"], dict):
                status = poll_json["data"].get("status")
            status_str = str(status).strip().upper() if status is not None else ""
            
            # 尝试获取进度
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
            except:
                prog_val = None

            prog_disp = int(prog_val) if prog_val is not None else str(progress)
            log_key = (status_str, prog_disp)
            if log_key != last_log_key:
                print(f"[Shaobkj-Sora] 任务ID: {task_id} | 状态: {status} | 进度: {progress}")
                last_log_key = log_key
            
            def find_video_url(obj):
                if isinstance(obj, dict):
                    if "video_url" in obj and obj["video_url"]:
                        return str(obj["video_url"])
                    if "url" in obj and obj["url"]:
                        u = str(obj["url"])
                        if u.startswith("http"):
                            return u
                    for v in obj.values():
                        res = find_video_url(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = find_video_url(item)
                        if res:
                            return res
                return None

            done_statuses = {"SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED", "DONE"}
            failed_statuses = {"FAILED", "FAIL", "ERROR", "FAILURE", "CANCELED", "CANCELLED"}
            video_url = find_video_url(poll_json)
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
                    print(f"[Shaobkj-Sora] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
                    if error_message:
                        raise RuntimeError(f"任务失败，{error_message}")
                    raise RuntimeError(f"任务失败({fail_reason})")
                if inner_err:
                    print(f"[Shaobkj-Sora] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
                    if error_message:
                        raise RuntimeError(f"任务失败，{error_message}")
                    raise RuntimeError(f"任务失败({inner_err})")
                print(f"[Shaobkj-Sora] 任务失败详情: {json.dumps(poll_json, ensure_ascii=False)}")
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
                print(f"[Shaobkj-Sora] 生成成功但未找到视频链接详情: {json.dumps(poll_json, ensure_ascii=False)}")
                raise RuntimeError("生成成功但未找到视频链接")
            
            if prog_val is not None:
                pbar_val = min(90, 40 + int(float(prog_val) * 0.5))
                pbar.update_absolute(pbar_val)
            else:
                max_attempts = 120 if int(等待时间) == 0 else max(1, int(timeout_val / max(1, poll_interval)))
                pbar_val = min(80, 40 + (attempts * 40 // max_attempts))
                pbar.update_absolute(pbar_val)
            
    def download_video(self, video_url, full_response, session=None, headers=None, api_origin="", timeout_budget=None):
        print(f"[Shaobkj-Sora] Downloading video from {video_url}...")
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
            
            # 将视频保存到临时文件以便加载
            filename = f"sora_{int(time.time())}_{random.randint(1000,9999)}.mp4"
            output_dir = folder_paths.get_output_directory()
            file_path = os.path.join(output_dir, filename)
            
            with open(file_path, "wb") as f:
                for chunk in v_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[Shaobkj-Sora] Video saved to {file_path}")
            
            video_obj = InputImpl.VideoFromFile(file_path)
            return (video_obj, json.dumps(full_response, ensure_ascii=False))
            
        except Exception as e:
            print(f"[Shaobkj-Sora] 加载视频帧失败: {e}")
            raise
