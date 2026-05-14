import base64
import io
import os
import json
import re
import time
import traceback
import random
import torch
import requests
from urllib.parse import urlparse
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
                "模型选择": (["gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"], {"default": "gemini-2.5-flash", "tooltip": "模型选择；推荐：gemini-2.5-flash"}),
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
        # Implementation of LLM Logic
        return self._execute_llm(API密钥, API地址, 模型选择, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置, 等待时间, seed, **kwargs)

    def _execute_llm(self, API密钥, API地址, 模型选择, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置, 等待时间, seed, **kwargs):
        api_key = API密钥
        if not api_key:
            raise ValueError("API Key is required.")

        base_origin = str(API地址).rstrip("/")
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

        # ... (Rest of the LLM logic is reused, we can extract this common logic if needed, but for now I will duplicate the core request logic to support both nodes) ...
        # For brevity and safety, I will keep the original implementation here and just call it.
        # However, since I cannot easily call the method from another class without inheritance or a helper, 
        # I will refactor the core request logic into a standalone function or keep it inside.
        
        # To avoid massive code duplication, let's just use the logic I read previously.
        # Since I'm rewriting the file, I'll include the full logic.
        
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        # ... (Logic for Gemini 2.5 Flash and others) ...
        
        # Actually, let's implement the full logic cleanly.
        
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(int(等待时间))

        # Construct Prompt
        system_prompt = 系统指令.strip() if isinstance(系统指令, str) else ""
        user_prompt = 用户输入.strip()
        
        # Special handling for Gemini 2.5 Flash (often doesn't support systemInstruction field well in some proxy/versions, so prepending is safer)
        # But for 3-pro-preview, systemInstruction is supported.
        # Let's use the standard payload structure.
        
        parts = [{"text": user_prompt}]
        append_images(parts)
        
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "topP": topP,
                "seed": int(seed) if int(seed) >= 0 else 0
            }
        }
        
        if system_prompt:
             payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        if 思考模式:
             payload["generationConfig"]["thinkingConfig"] = {
                "includeThoughts": True,
                "thinkingBudget": 思考预算
            }

        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
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
                return (f"API Error {response.status_code}: {response.text}", str(response.text))

            res_json = response.json()
            
            generated_text = ""
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                candidate = res_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]
            
            if not generated_text:
                generated_text = "No text response generated."
                
            return (generated_text, json.dumps(res_json, ensure_ascii=False))

        except Exception as e:
            return (f"Error: {str(e)}", str(traceback.format_exc()))


class Shaobkj_LLM_Test_API(Shaobkj_LLM_App):
    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型名称": ("STRING", {"default": "gemini-2.5-flash", "multiline": False, "tooltip": "自定义模型 ID；推荐：gemini-2.5-flash"}),
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

    def run_llm(self, API密钥, API地址, 模型名称, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置=1280, 等待时间=180, seed=0, **kwargs):
        return self._execute_llm(API密钥, API地址, 模型名称, 使用系统代理, 系统指令, 用户输入, 思考模式, 思考预算, temperature, topP, 输入图像_长边设置, 等待时间, seed, **kwargs)


class Shaobkj_Media_Reverse_Prompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型名称": ("STRING", {"default": "gemini-2.5-pro", "multiline": False, "tooltip": "自定义模型 ID；推荐：gemini-2.5-pro"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "提示词": ("STRING", {"default": "自动根据图片或视频输入使用专业反推提示词；如需自定义可直接覆盖", "multiline": True, "tooltip": "可选自定义反推要求；留空时自动使用专业反推提示词"}),
                "等待时间": ("INT", {"default": 180, "min": 0, "max": 1000000, "tooltip": "请求超时时间(秒)；推荐：180"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "随机种子；推荐：0"}),
            },
            "optional": {
                "图片": ("IMAGE", {"tooltip": "图片输入；可选"}),
                "视频": ("VIDEO", {"tooltip": "视频输入；可选"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("反推文本", "API响应")
    FUNCTION = "run"
    CATEGORY = "🤖shaobkj-APIbox"

    def _read_video_path(self, path):
        path = os.fspath(path).strip() if path is not None else ""
        if path and os.path.isfile(path):
            with open(path, "rb") as f:
                return f.read(), os.path.basename(path)
        return None, None

    def _video_file_to_bytes(self, video_input):
        if video_input is None:
            return None, None

        if isinstance(video_input, (list, tuple)):
            for item in video_input:
                data, filename = self._video_file_to_bytes(item)
                if data:
                    return data, filename
            return None, None

        get_stream = getattr(video_input, "get_stream_source", None)
        if callable(get_stream):
            try:
                source = get_stream()
                if isinstance(source, (str, os.PathLike)):
                    data, filename = self._read_video_path(source)
                    if data:
                        return data, filename
                elif isinstance(source, io.BytesIO):
                    source.seek(0)
                    data = source.read()
                    if data:
                        return data, f"reverse_video_{abs(hash(data)) % 10**10}.mp4"
                elif hasattr(source, "read"):
                    if hasattr(source, "seek"):
                        source.seek(0)
                    data = source.read()
                    if data:
                        return data, f"reverse_video_{abs(hash(data)) % 10**10}.mp4"
            except Exception as e:
                print(f"[Shaobkj-反推] 读取视频流失败: {e}")

        save_to = getattr(video_input, "save_to", None)
        if callable(save_to):
            try:
                buffer = io.BytesIO()
                save_to(buffer)
                buffer.seek(0)
                data = buffer.read()
                if data:
                    return data, f"reverse_video_{abs(hash(data)) % 10**10}.mp4"
            except Exception as e:
                print(f"[Shaobkj-反推] 导出视频失败: {e}")

        if isinstance(video_input, (str, os.PathLike)):
            return self._read_video_path(video_input)

        if isinstance(video_input, dict):
            for key in ("video", "VIDEO", "path", "file", "file_path", "filename"):
                value = video_input.get(key)
                if value is None:
                    continue
                data, filename = self._video_file_to_bytes(value)
                if data:
                    return data, filename

        for attr in ("path", "file_path", "filename", "_VideoFromFile__file"):
            value = getattr(video_input, attr, None)
            if value is None:
                continue
            data, filename = self._video_file_to_bytes(value)
            if data:
                return data, filename

        print(f"[Shaobkj-反推] 未识别的视频输入类型: {type(video_input)}")
        return None, None

    def _upload_video_to_gemini(self, session, base_origin, api_key, video_input, proxies=None):
        file_content, filename = self._video_file_to_bytes(video_input)
        if not file_content:
            return None
        if not filename:
            filename = f"reverse_video_{abs(hash(file_content)) % 10**10}.mp4"

        upload_url = f"{base_origin}/upload/v1beta/files"
        metadata_headers = {
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(file_content)),
            "X-Goog-Upload-Header-Content-Type": "video/mp4",
            "Content-Type": "application/json",
        }
        metadata = {"file": {"display_name": filename}}
        start_response = session.post(
            upload_url,
            headers=metadata_headers,
            json=metadata,
            timeout=120,
            proxies=proxies,
            verify=False,
        )
        start_response.raise_for_status()
        resumable_url = start_response.headers.get("x-goog-upload-url")
        if not resumable_url:
            raise RuntimeError(f"Gemini 视频上传初始化失败: {start_response.text}")

        upload_response = session.post(
            resumable_url,
            headers={
                "Content-Length": str(len(file_content)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=file_content,
            timeout=120,
            proxies=proxies,
            verify=False,
        )
        upload_response.raise_for_status()
        file_info = upload_response.json().get("file", {})
        file_uri = file_info.get("uri")
        if not file_uri:
            raise RuntimeError(f"Gemini 视频上传完成但未返回 file.uri: {upload_response.text}")

        file_name = file_info.get("name")
        state = file_info.get("state")
        if file_name and state and state != "ACTIVE":
            file_url = f"{base_origin}/v1beta/{file_name}"
            for _ in range(60):
                time.sleep(2)
                status_response = session.get(
                    file_url,
                    headers={"x-goog-api-key": api_key},
                    timeout=30,
                    proxies=proxies,
                    verify=False,
                )
                status_response.raise_for_status()
                file_info = status_response.json()
                state = file_info.get("state")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise RuntimeError(f"Gemini 视频处理失败: {status_response.text}")
            else:
                raise RuntimeError("Gemini 视频处理超时，请换短一点的视频或加大等待时间。")

        return file_uri

    def _video_inline_part(self, video_input):
        file_content, _ = self._video_file_to_bytes(video_input)
        if not file_content:
            raise ValueError("视频读取失败，当前视频输入无法转换为可用文件。")
        return {
            "inline_data": {
                "mime_type": "video/mp4",
                "data": base64.b64encode(file_content).decode("utf-8"),
            }
        }

    def _image_parts(self, image_tensor, long_side):
        parts = []
        if image_tensor is None:
            return parts
        if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() == 4:
            for i in range(image_tensor.shape[0]):
                pil_img = tensor_to_pil(image_tensor[i])
                b64_str, _ = resize_and_encode_image(pil_img, long_side)
                if b64_str:
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})
        else:
            pil_img = tensor_to_pil(image_tensor)
            b64_str, _ = resize_and_encode_image(pil_img, long_side)
            if b64_str:
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64_str}})
        return parts

    def _image_prompt(self):
        return (
            "你是一位资深视觉导演、商业摄影策划与提示词工程师。"
            "请基于输入图片进行专业反推，输出一段可直接用于图像生成或图像编辑模型的高质量中文提示词。"
            "要求准确描述主体、构图、镜头视角、景别、姿态、服装或材质、光线、色彩、背景环境、氛围、风格、细节质感与清晰度。"
            "若能识别出适合复现的摄影语言、设计语言或艺术风格，也要自然写入。"
            "不要解释，不要分点，不要写分析过程，只输出一段精炼、专业、可直接使用的中文提示词。"
        )

    def _video_prompt(self):
        return (
            "你是一位资深分镜导演、摄影指导、剪辑策划与视频提示词工程师。"
            "请基于输入视频进行专业反推，输出一段可直接用于视频生成或视频编辑模型的高质量中文提示词。"
            "要求准确描述主体、场景、镜头运动、景别变化、人物或物体动作、节奏、转场感、光线、色彩、氛围、画面风格与关键时间顺序。"
            "如果视频具有明显的运镜方式、叙事结构或风格特征，也要自然融入提示词中。"
            "不要解释，不要分点，不要写分析过程，只输出一段精炼、专业、可直接使用的中文提示词。"
        )

    def _build_prompt(self, custom_prompt, has_image, has_video):
        text = str(custom_prompt or "").strip()
        if text:
            return text
        if has_video:
            return self._video_prompt()
        if has_image:
            return self._image_prompt()
        return ""

    def run(self, API密钥, API地址, 模型名称, 使用系统代理, 提示词, 等待时间, seed, 图片=None, 视频=None, **kwargs):
        if not API密钥:
            raise ValueError("API Key is required.")

        has_image = 图片 is not None
        has_video = 视频 is not None
        if not has_image and not has_video:
            raise ValueError("图片或视频至少需要提供一个。")

        base_origin = str(API地址).rstrip("/")
        if base_origin.endswith("/v1"):
            base_origin = base_origin[:-3]
        model = str(模型名称).strip() or "gemini-2.5-pro"
        prompt_text = self._build_prompt(提示词, has_image, has_video)
        long_side_val = int(kwargs.get("输入图像-长边设置", 1280))
        timeout_val = None if int(等待时间) == 0 else int(等待时间)

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        submit_timeout = build_submit_timeout(timeout_val if timeout_val is not None else 180)

        content_parts = [{"text": prompt_text}]
        if has_video:
            content_parts.append(self._video_inline_part(视频))
        elif has_image:
            content_parts.extend(self._image_parts(图片, long_side_val))

        payload = {
            "contents": [{"role": "user", "parts": content_parts}],
            "generationConfig": {
                "temperature": 0.2,
                "seed": int(seed) if int(seed) >= 0 else 0,
            },
        }

        headers = {"Content-Type": "application/json", "x-goog-api-key": API密钥}
        url = f"{base_origin}/v1beta/models/{model}:generateContent"

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
                return (f"API Error {response.status_code}: {response.text}", str(response.text))
            res_json = response.json()
            generated_text = ""
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                candidate = res_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]
            return ((generated_text or "No text response generated."), json.dumps(res_json, ensure_ascii=False))
        except Exception as e:
            return (f"Error: {str(e)}", str(traceback.format_exc()))


class Shaobkj_NanoBanana_Prompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "服务端 API Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址"}),
                "模型选择": (["gemini-2.5-flash", "gemini-3.1-pro-preview", "gemini-3-flash-preview"], {"default": "gemini-2.5-flash", "tooltip": "模型选择"}),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理"}),
                "用户输入": ("STRING", {"default": "", "multiline": True, "tooltip": "描述你想生成的画面，例如：制作一张xx品牌香水的展示海报"}),
                "任务类型": (["生成模式 (Generation)", "编辑模式 (Editing)"], {"default": "生成模式 (Generation)", "tooltip": "选择生成提示词还是编辑指令"}),
                "场景风格": (
                    ["Editorial (杂志大片)", "Cinematic (电影质感)", "Product Shot (产品特写)", "Minimalist (极简主义)", "None (不指定)"], 
                    {"default": "Editorial (杂志大片)", "tooltip": "仅在生成模式下生效"}
                ),
                "品牌名称": ("STRING", {"default": "", "multiline": False, "tooltip": "可选：替换提示词中的 [BRAND]"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "generate_prompt"
    CATEGORY = "🤖shaobkj-APIbox"

    def generate_prompt(self, API密钥, API地址, 用户输入, 任务类型, 模型选择, 场景风格, 品牌名称, 使用系统代理, seed, **kwargs):
        # 1. Select System Prompt based on Task Type
        is_editing_mode = "Editing" in 任务类型
        quality_boosters = ["高画质", "细节清晰", "真实光影", "电影级质感", "8k分辨率"]
        random_seed = int(seed) if int(seed) >= 0 else 0

        def ensure_two_quality_boosters(text):
            existing = [w for w in quality_boosters if w in text]
            if len(existing) >= 2:
                return text
            rng = random.Random(random_seed + len(text))
            pool = [w for w in quality_boosters if w not in existing]
            rng.shuffle(pool)
            selected = existing + pool[: max(0, 2 - len(existing))]
            selected = selected[:2]
            if not selected:
                return text
            suffix = "，".join(selected)
            if text.strip().endswith(("。", "！", "？")):
                return f"{text}{suffix}。"
            return f"{text}。{suffix}。"

        if "Editing" in 任务类型:
            system_prompt = """gemini-3-pro-image-preview多图编辑指令优化
# Role
你是一位精通 "gemini-3-pro-image-preview" 图像编辑模型的提示词专家。你的核心能力是理解用户的多图参考意图，生成精准、克制且高质量的中文编辑指令。

# Task
根据用户的自然语言描述，生成适用于 gemini-3-pro-image-preview 的中文 Prompt。

# Constraints & Rules
1. 中文输出：所有的 Prompt 必须使用简体中文。
2. 命令式句式：使用直接、果断的动作动词开头（如：替换、修改、融合、保持、提取）。
3. 多图逻辑处理：
- 用户可能提供多张参考图（参考图1、参考图2...）和一张目标图。
- 必须清晰界定“参考源”与“目标对象”的关系（例如：“提取参考图1的风格赋予目标图”、“将参考图2的物体融合进目标图”）。
4. 克制联想（高保真）：
- 严禁随意添加用户未提及的物体或复杂背景。
- 仅在材质、光影、清晰度这三个维度进行必要的专业术语补全，以保证生成质量。
5. 画质增强：在 Prompt 末尾追加 2 个通用的画质增强词。

# Output Format
直接输出 Prompt 内容，不包含任何解释、前言或额外符号。

# Workflow
1. 分析用户输入，识别涉及的图片数量及各自的角色（参考 vs 目标）。
2. 提取核心动作（换脸、换装、风格迁移、背景替换等）。
3. 编写中文指令，确保指代明确（如“将参考图A的...”）。
4. 检查是否过度联想，删除多余的形容词。
5. 追加画质词。

# Examples
User: 把这张图里的人换成这几张参考图里的那个模特，衣服不要变。
Output: 保持目标图中的服装和背景不变，将人物面部和体型替换为参考图中的模特特征。确保面部光影与原环境完美融合，皮肤质感真实自然。高画质，细节清晰。

User: 用参考图1的配色和参考图2的画风，重绘这张草图。
Output: 提取参考图1的色彩方案，结合参考图2的绘画风格，对目标草图进行上色和细化重绘。保持草图原有的构图和轮廓，色彩过渡自然。杰作，高分辨率。

User: 给这个杯子加个盖子。
Output: 在杯子上方添加一个材质匹配的盖子。确保盖子的透视角度、光影反射与杯身一致。真实感，8k画质。

# Quality Boosters (Randomly pick 2)
(高画质, 细节清晰, 真实光影, 电影级质感, 8k分辨率)"""
            user_content = f"用户需求：{用户输入}"
        else:
            style_guide = ""
            if "Editorial" in 场景风格:
                style_guide = "Style: High-end fashion editorial, Vogue-style photography, studio lighting."
            elif "Cinematic" in 场景风格:
                style_guide = "Style: Cinematic movie still, dramatic lighting, anamorphic lens flare, color graded."
            elif "Product" in 场景风格:
                style_guide = "Style: Professional product photography, macro details, sharp focus, commercial lighting."
            elif "Minimalist" in 场景风格:
                style_guide = "Style: Minimalist composition, clean lines, negative space, soft pastel colors."

            system_prompt = f"""You are an expert prompt engineer for Nano Banana Pro (Gemini 3 Pro Image).
Your task is to convert the user's simple description into a high-quality, hyper-realistic prompt following this structure:
[Concept & Vibe] -> [Composition & Angle] -> [Subject Details] -> [Lighting & Atmosphere] -> [Texture & Realism].

Reference Example:
"A hyper-realistic editorial concept for a collaboration between [BRAND] and [MAGAZINE BRAND]. Square 1:1 composition, shot in a sleek Parisian interior with marble floors and tall windows, golden afternoon light illuminating the scene. A single model in a couture gown poses gracefully beside a realistically sized [BRAND] perfume bottle with the [BRAND] logo clearly visible placed on a marble pedestal. Ultra-refined textures, cinematic realism, Vogue-style photography."

Instructions:
1. {style_guide}
2. If the user provides a brand name, use it naturally.
3. Output ONLY the final English prompt.
4. Enhance details for realism (8k, detailed texture, volumetric lighting)."""
            user_content = f"User Description: {用户输入}\nBrand Name: {品牌名称}"

        # 2. Call LLM (Reusing Shaobkj_LLM_App logic wrapper or simple request)
        # To avoid dependency issues, we implement a lightweight request here.
        
        api_key = API密钥
        if not api_key:
            return ("Error: API Key is required.",)

        base_origin = str(API地址).rstrip("/")
        if base_origin.endswith("/v1"):
            base_origin = base_origin[:-3]
        
        model = str(模型选择 or "gemini-2.5-flash")
        url = f"{base_origin}/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {"temperature": 0.7, "seed": int(seed)}
        }
        
        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))
        
        try:
            response = post_json_with_retry(
                session, url, headers=headers, payload=payload, timeout=60, proxies=proxies, verify=False
            )
            if response.status_code != 200:
                return (f"Error: {response.text}",)
            
            res_json = response.json()
            generated_text = ""
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                candidate = res_json["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            generated_text += part["text"]
            
            # Post-processing: Replace [BRAND] placeholder if needed (though LLM should have handled it)
            if 品牌名称 and "[BRAND]" in generated_text:
                generated_text = generated_text.replace("[BRAND]", 品牌名称)
            if is_editing_mode:
                generated_text = ensure_two_quality_boosters(generated_text)
                
            return (generated_text.strip(),)
            
        except Exception as e:
            return (f"Error: {str(e)}",)
