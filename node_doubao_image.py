import base64
import io
import json
from urllib.parse import urlparse

from PIL import Image
from comfy.utils import ProgressBar
import torch


from .shaobkj_shared import (
    create_requests_session,
    disable_insecure_request_warnings,
    get_config_value,
    pil_to_tensor,
    post_json_with_retry,
    resize_pil_long_side,
    sanitize_text,
    tensor_to_pil,
)


class Shaobkj_Doubao_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        api_key_default = get_config_value("API_KEY", "SHAOBKJ_API_KEY", "")
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "生成内容描述，支持中英文；推荐：简洁明确"}),
                "API密钥": ("STRING", {"default": api_key_default, "multiline": False, "tooltip": "云雾 API Key；推荐：填写有效 Key"}),
                "API地址": ("STRING", {"default": "https://yhmx.work", "multiline": False, "tooltip": "API 基础地址；推荐：https://yhmx.work"}),
                "模型选择": (
                    [
                        "doubao-seedream-5-0-260128",
                        "doubao-seedream-4-0-250828",
                        "doubao-seedream-4-5-251128",
                    ],
                    {"default": "doubao-seedream-5-0-260128", "tooltip": "默认模型；推荐：doubao-seedream-5-0-260128"},
                ),
                "使用系统代理": ("BOOLEAN", {"default": True, "tooltip": "是否使用系统代理；推荐：开启"}),
                "尺寸": (
                    ["1K", "2K", "4K", "1024x1024", "1536x1536", "2048x2048"],
                    {"default": "2K", "tooltip": "输出尺寸；推荐：2K"},
                ),
                "返回格式": (["url", "b64_json"], {"default": "url", "tooltip": "接口返回格式；推荐：url"}),
                "组图模式": (["disabled", "auto"], {"default": "disabled", "tooltip": "是否启用组图输出；推荐：disabled"}),
                "最多出图数": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1, "tooltip": "组图模式下最多出图数量；推荐：3"}),
                "是否水印": ("BOOLEAN", {"default": False, "tooltip": "是否添加 AI 水印；推荐：关闭"}),
                "输入图像-长边设置": (["1024", "1280", "1536"], {"default": "1280", "tooltip": "输入图像长边缩放；推荐：1280"}),
                "等待时间": ("INT", {"default": 180, "min": 1, "max": 1000000, "tooltip": "请求等待时间(秒)；推荐：180"}),
            },
            "optional": {
                "参考图1": ("IMAGE", {"tooltip": "输入图像1；推荐：可选"}),
                "参考图2": ("IMAGE", {"tooltip": "输入图像2；推荐：可选"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "API响应")
    FUNCTION = "generate_image"
    CATEGORY = "🤖shaobkj-APlbox"

    def generate_image(self, 提示词, API密钥, API地址, 模型选择, 使用系统代理, 尺寸, 返回格式, 组图模式, 最多出图数, 是否水印, 等待时间, 参考图1=None, 参考图2=None, **kwargs):
        if not API密钥:
            raise ValueError("API Key is required.")

        prompt = str(提示词 or "").strip()
        if not prompt:
            raise ValueError("提示词不能为空。")

        base_url = str(API地址).rstrip("/")
        url = f"{base_url}/v1/images/generations"
        api_origin = urlparse(base_url).netloc
        timeout_val = int(等待时间)
        long_side_limit = int(kwargs.get("输入图像-长边设置", 1280))

        pbar = ProgressBar(100)
        pbar.update_absolute(0)

        headers = {
            "Authorization": f"Bearer {API密钥}",
            "Content-Type": "application/json",
        }

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

        images_payload = []
        for tensor in reference_images:
            pil_img = tensor_to_pil(tensor)
            pil_img = resize_pil_long_side(pil_img, long_side_limit)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=95)
            image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_payload.append(f"data:image/jpeg;base64,{image_b64}")

        payload = {
            "model": str(模型选择),
            "prompt": prompt,
            "response_format": str(返回格式),
            "size": str(尺寸),
            "watermark": bool(是否水印),
            "stream": False,
        }
        if images_payload:
            payload["image"] = images_payload
        if str(组图模式) == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {
                "max_images": int(最多出图数)
            }
        else:
            payload["sequential_image_generation"] = "disabled"

        disable_insecure_request_warnings()
        session, proxies = create_requests_session(bool(使用系统代理))

        def decode_b64_image(b64_str):
            try:
                image_data = base64.b64decode(b64_str)
                image = Image.open(io.BytesIO(image_data))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            except Exception:
                return None

        def download_url_image(image_url):
            try:
                image_headers = None
                if urlparse(str(image_url)).netloc == api_origin:
                    image_headers = {"Authorization": f"Bearer {API密钥}"}
                response = session.get(
                    image_url,
                    headers=image_headers,
                    verify=False,
                    timeout=timeout_val,
                    proxies=proxies,
                )
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return image
            except Exception:
                return None

        def extract_images(res_json):
            result_images = []
            if not isinstance(res_json, dict):
                return result_images
            data_list = res_json.get("data")
            if not isinstance(data_list, list):
                return result_images
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                image_obj = None
                if str(返回格式) == "b64_json" and item.get("b64_json"):
                    image_obj = decode_b64_image(item.get("b64_json"))
                elif isinstance(item.get("url"), str):
                    image_obj = download_url_image(item.get("url"))
                elif item.get("b64_json"):
                    image_obj = decode_b64_image(item.get("b64_json"))
                if image_obj is not None:
                    result_images.append(image_obj)
            return result_images

        try:
            pbar.update_absolute(20)
            response = post_json_with_retry(
                session,
                url,
                headers=headers,
                payload=payload,
                timeout=(10, timeout_val),
                proxies=proxies,
                verify=False,
            )
            response.raise_for_status()
            pbar.update_absolute(60)

            try:
                res_json = response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"接口返回不是有效 JSON: {exc}")

            result_images = extract_images(res_json)
            if not result_images:
                raise RuntimeError(f"未从接口响应中解析到图像: {sanitize_text(json.dumps(res_json, ensure_ascii=False))}")

            result_tensor = pil_to_tensor(result_images[0]) if len(result_images) == 1 else torch.cat([pil_to_tensor(img) for img in result_images], dim=0)

            first_image = result_images[0]
            width, height = first_image.size
            response_lines = [
                "状态: 成功",
                f"模型: {模型选择}",
                f"尺寸: {尺寸}",
                f"返回格式: {返回格式}",
                f"组图模式: {组图模式}",
                f"参考图数量: {len(reference_images)}",
                f"返回张数: {len(result_images)}",
                f"实际尺寸: {int(width)}x{int(height)}",
            ]
            pbar.update_absolute(100)
            return {"ui": {"images": []}, "result": (result_tensor, "\n".join(response_lines))}
        except Exception as exc:
            raise RuntimeError(f"请求失败: {exc}")

