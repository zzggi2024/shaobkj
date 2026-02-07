import json
import re
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
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False}),
                "模型名称": ("STRING", {"default": "gemini-3-pro-preview", "multiline": False}),
                "使用系统代理": ("BOOLEAN", {"default": True}),
                "系统指令": ("STRING", {"default": "你是高效的AI提示词生成大师。请根据用户输入生成可直接执行的方案或内容，结构清晰，直接输出提示词，不要有任何废话。", "multiline": True}),
                "用户输入": ("STRING", {"default": "", "multiline": True}),
                "思考模式": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
                "思考预算": ("INT", {"default": 10240, "min": 1024, "max": 65536, "step": 1024}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "topP": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "请求超时时间(秒)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "API申请地址": ("STRING", {"default": "https://yhmx.work/login?expired=true", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本内容",)
    FUNCTION = "run_llm"
    CATEGORY = "🤖shaobkj-APIbox"

    def run_llm(self, API密钥, API地址, 模型名称, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置=1280, 等待时间=180, seed=0, **kwargs):
        api_key = API密钥
        if not api_key:
            raise ValueError("API Key is required.")

        base_origin = str(API地址).rstrip("/")
        # Remove /v1 suffix if present to avoid duplication if user provides full OpenAI-style base
        if base_origin.endswith("/v1"):
            base_origin = base_origin[:-3]
            
        model = 模型名称
        
        # Construct URL according to the documentation: 
        # POST /v1beta/models/{model}:streamGenerateContent
        url = f"{base_origin}/v1beta/models/{model}:streamGenerateContent"

        # Headers: Only x-goog-api-key as per project rules to improve proxy compatibility
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        # Construct Body
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

        long_side_val = int(kwargs.get("输入图像-长边设置", 输入图像_长边设置))
        image_inputs = []
        for k, v in kwargs.items():
            if k.startswith("image_") and v is not None:
                image_inputs.append((k, v))
        image_inputs.sort(key=lambda x: int(x[0].split("_")[1]))
        for _, tensor in image_inputs:
            try:
                pil_img = tensor_to_pil(tensor)
                b64_str, _ = resize_and_encode_image(pil_img, long_side_val)
                if b64_str:
                    payload["contents"][0]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_str
                        }
                    })
            except Exception:
                pass

        # Add system instruction if provided
        if 系统指令 and len(系统指令.strip()) > 0:
            payload["systemInstruction"] = {
                "parts": [
                    {"text": 系统指令}
                ]
            }

        # Add thinking config if enabled
        if 思考模式:
            payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": 思考预算
            }
            
        # Prepare network request
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(使用系统代理)
        
        timeout = build_submit_timeout(等待时间)
        
        try:
            print(f"[Shaobkj-LLM] Sending request to {url}...")
            response = post_json_with_retry(
                session,
                url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                proxies=proxies,
                max_retries=1
            )
            
            # Parse Response
            # streamGenerateContent returns a list of JSON objects (chunks)
            try:
                data = response.json()
            except Exception:
                # If json() fails, it might be NDJSON or raw text
                text_response = response.text
                # Try to parse as NDJSON
                data = []
                for line in text_response.splitlines():
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except Exception:
                            pass
                            
            # Aggregate text from all parts
            full_text = ""
            
            # Helper to extract text from a single chunk
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
                # Fallback for unexpected format
                full_text = str(data)

            if isinstance(full_text, str) and not full_text.strip():
                PromptServer.instance.send_sync(
                    "shaobkj.llm.warning",
                    {"message": "⚠️ 输出为空，请检查输入内容或接口返回。"}
                )
            return (full_text,)

        except Exception as e:
            error_msg = f"LLM Request Failed: {str(e)}"
            print(f"[Shaobkj-LLM] {error_msg}")
            return (error_msg,)
