import json
import re
import io
import base64
import traceback
import torch
import requests
from urllib.parse import urlparse
from server import PromptServer
from .shaobkj_shared import (
    get_config_value,
    post_json_with_retry,
    create_requests_session,
    disable_insecure_request_warnings,
    build_submit_timeout,
    resize_and_encode_image,
    resize_pil_long_side,
    tensor_to_pil
)

class Shaobkj_LLM_App:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (["gemini-2.5-flash", "gemini-3-pro-preview"], {"default": "gemini-2.5-flash", "tooltip": "模型选择；推荐：gemini-2.5-flash"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "系统指令": ("STRING", {"default": "你是高效的AI提示词生成大师。请根据用户输入生成可直接执行的方案或内容，结构清晰，直接输出提示词，不要有任何废话。", "multiline": True, "tooltip": "系统级指令；推荐：默认内容"}),
                "用户输入": ("STRING", {"default": "", "multiline": True, "tooltip": "用户输入内容；推荐：清晰具体"}),
                "思考模式": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "是否启用思考模式；推荐：关闭"}),
                "思考预算": ("INT", {"default": 10240, "min": 1024, "max": 65536, "step": 1024, "tooltip": "思考预算上限；推荐：10240"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "采样温度；推荐：0.7"}),
                "topP": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "核采样概率；推荐：0.95"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "请求超时时间(秒)；推荐：180"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False, "tooltip": "API 申请入口；推荐：默认地址"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("文本内容", "API响应")
    FUNCTION = "run_llm"
    CATEGORY = "🤖shaobkj-APIbox"

    def run_llm(self, API密钥, API地址, 模型选择, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置=1280, 等待时间=180, seed=0, **kwargs):
        api_key = API密钥
        if not api_key:
            raise ValueError("API Key is required.")

        base_origin = str(API地址).rstrip("/")
        # Remove /v1 suffix if present to avoid duplication if user provides full OpenAI-style base
        if base_origin.endswith("/v1"):
            base_origin = base_origin[:-3]
            
        model = 模型选择
        long_side_val = int(kwargs.get("输入图像-长边设置", 输入图像_长边设置))
        image_inputs = []
        for k, v in kwargs.items():
            if k.startswith("image_") and v is not None:
                image_inputs.append((k, v))
        image_inputs.sort(key=lambda x: int(x[0].split("_")[1]))

        def append_images(parts):
            for _, tensor in image_inputs:
                try:
                    if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
                        for i in range(tensor.shape[0]):
                            pil_img = tensor_to_pil(tensor[i])
                            b64_str, _ = resize_and_encode_image(pil_img, long_side_val)
                            if b64_str:
                                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})
                    else:
                        pil_img = tensor_to_pil(tensor)
                        b64_str, _ = resize_and_encode_image(pil_img, long_side_val)
                        if b64_str:
                            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})
                except Exception:
                    pass

        if model == "gemini-2.5-flash":
            system_prompt = 系统指令.strip() if isinstance(系统指令, str) else ""
            user_prompt = 用户输入.strip()
            prompt = (system_prompt + "\n\n" if system_prompt else "") + user_prompt
            url = f"{base_origin}/v1beta/models/{model}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            parts = [{"text": prompt}]
            append_images(parts)
            payload = {"contents": [{"role": "user", "parts": parts}]}
            safe_seed = int(seed)
            if safe_seed < 0:
                safe_seed = 0
            if safe_seed > 2147483647:
                safe_seed = safe_seed % 2147483647
            payload["generationConfig"] = {"seed": safe_seed}

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
                if isinstance(payload, dict) and "error" in payload:
                    err = payload["error"]
                    if isinstance(err, dict):
                        msg = err.get("message", "")
                        if "quota" in msg.lower() or "limit" in msg.lower():
                            print(f"[ComfyUI-shaobkj] Possible quota error: {msg}")

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
                    try:
                        err_msg = response.json()
                    except Exception:
                        err_msg = response.text
                    raise_if_quota_error(response.status_code, err_msg)
                    raise RuntimeError(f"API Error {response.status_code}: {err_msg}")
                try:
                    res_json = response.json()
                except (json.JSONDecodeError, ValueError) as e:
                    raw_text = response.text
                    if not raw_text or not raw_text.strip():
                        raise RuntimeError(f"API Error: Empty response body (HTTP {response.status_code})")
                    raise RuntimeError(f"Invalid JSON response from API: {e}")
                generated_text = ""
                if "candidates" in res_json and len(res_json["candidates"]) > 0:
                    candidate = res_json["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                generated_text += part["text"]
                if not generated_text:
                    generated_text = "No text response generated."
                api_resp_text = json.dumps(res_json, ensure_ascii=False)
                if not isinstance(api_resp_text, str):
                    api_resp_text = str(api_resp_text)
                if len(api_resp_text) > 8000:
                    api_resp_text = api_resp_text[:8000] + "...(truncated)"
                return (generated_text, api_resp_text)
            except Exception as e:
                error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                print(f"[ComfyUI-shaobkj] Inference Error: {error_msg}")
                return (f"Inference Failed: {str(e)}", error_msg)

        url = f"{base_origin}/v1beta/models/{model}:streamGenerateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": 用户输入}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": topP,
            }
        }
        append_images(payload["contents"][0]["parts"])
        if 系统指令 and len(系统指令.strip()) > 0:
            payload["systemInstruction"] = {"parts": [{"text": 系统指令}]}
        if 思考模式:
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": 思考预算
            }
        safe_seed = int(seed)
        if safe_seed < 0:
            safe_seed = 0
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647
        payload["generationConfig"]["seed"] = safe_seed
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(使用系统代理)
        timeout = build_submit_timeout(等待时间)
        try:
            response = post_json_with_retry(
                session,
                url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                proxies=proxies,
                max_retries=1
            )
            try:
                data = response.json()
                api_resp_text = json.dumps(data, ensure_ascii=False)
            except Exception:
                text_response = response.text
                data = []
                for line in text_response.splitlines():
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except Exception:
                            pass
                api_resp_text = text_response

            full_text = ""

            def extract_text(chunk):
                text = ""
                if "candidates" in chunk:
                    for candidate in chunk["candidates"]:
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    text += part["text"]
                return text

            if isinstance(data, list):
                for chunk in data:
                    full_text += extract_text(chunk)
            elif isinstance(data, dict):
                full_text = extract_text(data)
            else:
                full_text = str(data)

            if isinstance(full_text, str) and not full_text.strip():
                PromptServer.instance.send_sync(
                    "shaobkj.llm.warning",
                    {"message": "⚠️ 输出为空，请检查输入内容或接口返回。"}
                )
            return (full_text, api_resp_text if isinstance(api_resp_text, str) else str(api_resp_text))
        except Exception as e:
            error_msg = f"LLM Request Failed: {str(e)}"
            print(f"[Shaobkj-LLM] {error_msg}")
            return (error_msg, error_msg)
