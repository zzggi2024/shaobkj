import os
import re
from unittest.mock import patch

import folder_paths
import torch

from .shaobkj_shared import pil_to_tensor, tensor_to_pil

FL2_MODEL_REPOS = {
    "base": "microsoft/Florence-2-base",
    "base-ft": "microsoft/Florence-2-base-ft",
    "large": "microsoft/Florence-2-large",
    "large-ft": "microsoft/Florence-2-large-ft",
    "DocVQA": "HuggingFaceM4/Florence-2-DocVQA",
    "SD3-Captioner": "gokaygokay/Florence-2-SD3-Captioner",
    "base-PromptGen": "MiaoshouAI/Florence-2-base-PromptGen",
    "CogFlorence-2-Large-Freeze": "thwri/CogFlorence-2-Large-Freeze",
    "CogFlorence-2.1-Large": "thwri/CogFlorence-2.1-Large",
    "base-PromptGen-v1.5": "MiaoshouAI/Florence-2-base-PromptGen-v1.5",
    "large-PromptGen-v1.5": "MiaoshouAI/Florence-2-large-PromptGen-v1.5",
    "base-PromptGen-v2.0": "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
    "large-PromptGen-v2.0": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    "Florence-2-Flux": "gokaygokay/Florence-2-Flux",
    "Florence-2-Flux-Large": "gokaygokay/Florence-2-Flux-Large",
}

TASK_TOKEN_MAP = {
    "caption": "<CAPTION>",
    "detailed caption": "<DETAILED_CAPTION>",
    "more detailed caption": "<MORE_DETAILED_CAPTION>",
    "description": "<DESCRIPTION>",
    "generate tags(PromptGen 1.5)": "<GENERATE_TAGS>",
    "mixed caption(PromptGen 1.5)": "<MIXED_CAPTION>",
    "mixed caption plus(PromptGen 2.0)": "<MIXED_CAPTION_PLUS>",
    "analyze(PromptGen 2.0)": "<<ANALYZE>>",
    "object detection": "<OD>",
    "dense region caption": "<DENSE_REGION_CAPTION>",
    "region proposal": "<REGION_PROPOSAL>",
    "region proposal (mask)": "<REGION_PROPOSAL>",
    "caption to phrase grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "open vocabulary detection": "<OPEN_VOCABULARY_DETECTION>",
    "region to category": "<REGION_TO_CATEGORY>",
    "region to description": "<REGION_TO_DESCRIPTION>",
    "OCR": "<OCR>",
    "OCR with region": "<OCR_WITH_REGION>",
}


def _fixed_get_imports(filename):
    from transformers.dynamic_module_utils import get_imports as dynamic_get_imports

    if os.path.basename(filename) != "modeling_florence2.py":
        return dynamic_get_imports(filename)
    imports = dynamic_get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def _candidate_dtypes():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return [torch.bfloat16, torch.float16, torch.float32]
        return [torch.float16, torch.float32]
    return [torch.float32]


def _candidate_attn():
    if torch.cuda.is_available():
        return ["sdpa", "eager"]
    return ["eager"]


def _remove_angle_brackets(text):
    return re.sub(r"<[^>]*>", "", str(text))


def _extract_text(result):
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "labels" in result and isinstance(result["labels"], list):
            labels = [str(x).replace("</s>", "").strip() for x in result["labels"] if str(x).strip()]
            if labels:
                return ", ".join(labels)
        for value in result.values():
            text = _extract_text(value)
            if text:
                return text
    if isinstance(result, list):
        for item in result:
            text = _extract_text(item)
            if text:
                return text
    return str(result)


def _load_florence_model(version):
    try:
        from transformers.dynamic_module_utils import get_imports as original_get_imports
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as e:
        raise RuntimeError(f"缺少依赖，请安装 transformers 与 huggingface_hub: {e}") from e

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model_root = os.path.join(folder_paths.models_dir, "florence2")
    os.makedirs(model_root, exist_ok=True)
    model_path = os.path.join(model_root, version)

    if not os.path.exists(model_path):
        snapshot_download(
            repo_id=FL2_MODEL_REPOS[version],
            local_dir=model_path,
            ignore_patterns=["*.md", "*.txt"],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_error = None

    for dtype in _candidate_dtypes():
        for attn in _candidate_attn():
            try:
                def safe_get_imports(filename):
                    if os.path.basename(filename) != "modeling_florence2.py":
                        return original_get_imports(filename)
                    imports = original_get_imports(filename)
                    if "flash_attn" in imports:
                        imports.remove("flash_attn")
                    return imports

                with patch("transformers.dynamic_module_utils.get_imports", safe_get_imports):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        attn_implementation=attn,
                    )
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model = model.to(device)
                model.eval()
                return {
                    "model": model,
                    "processor": processor,
                    "version": version,
                    "device": device,
                    "dtype": dtype,
                    "attn": attn,
                }
            except Exception as e:
                last_error = e
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    raise RuntimeError(f"加载 Florence2 模型失败: {last_error}")


def _run_florence(model_bundle, image, task, text_input, max_new_tokens, num_beams, do_sample):
    if not isinstance(model_bundle, dict):
        raise RuntimeError("Florence2模型对象无效，请重新连接‘🤖加载Florence2模型(急速)’输出。")

    task_token = TASK_TOKEN_MAP.get(task, "<MORE_DETAILED_CAPTION>")
    prompt = task_token if not text_input else f"{task_token}{text_input}"
    model = model_bundle.get("model")
    processor = model_bundle.get("processor")
    if model is None or processor is None:
        raise RuntimeError("Florence2模型缺少 model/processor，请重新加载模型。")

    device = model_bundle.get("device")
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = model_bundle.get("dtype")
    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except Exception:
            dtype = torch.float32

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=dtype):
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=int(max_new_tokens),
                    early_stopping=False,
                    do_sample=bool(do_sample),
                    num_beams=int(num_beams),
                    use_cache=True,
                )
        else:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=int(max_new_tokens),
                early_stopping=False,
                do_sample=bool(do_sample),
                num_beams=int(num_beams),
                use_cache=True,
            )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    try:
        return processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(image.width, image.height),
        )
    except Exception:
        return generated_text


class Shaobkj_Load_Florence2_Model:
    def __init__(self):
        self.model_bundle = None
        self.version = None

    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(FL2_MODEL_REPOS.keys())
        return {
            "required": {
                "模型版本": (model_list, {"default": "large-PromptGen-v2.0"}),
            },
        }

    RETURN_TYPES = ("FLORENCE2",)
    RETURN_NAMES = ("Florence2模型",)
    FUNCTION = "load"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def load(self, 模型版本):
        if self.model_bundle is None or self.version != 模型版本:
            self.model_bundle = _load_florence_model(模型版本)
            self.version = 模型版本
        return (self.model_bundle,)


class Shaobkj_Florence2_Fast_Prompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        caption_task_list = list(TASK_TOKEN_MAP.keys())
        return {
            "required": {
                "Florence2模型": ("FLORENCE2",),
                "图像": ("IMAGE",),
                "任务": (caption_task_list, {"default": "more detailed caption"}),
                "文本输入": ("STRING", {"default": ""}),
                "最大新token": ("INT", {"default": 1024, "min": 1, "step": 1}),
                "束搜索数": ("INT", {"default": 3, "min": 1, "step": 1}),
                "采样": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("提示词", "预览图")
    FUNCTION = "run"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def run(self, Florence2模型, 图像, 任务, 文本输入, 最大新token, 束搜索数, 采样):
        img = tensor_to_pil(图像[0]).convert("RGB")
        result = _run_florence(
            Florence2模型,
            img,
            任务,
            文本输入,
            最大新token,
            束搜索数,
            采样,
        )
        text = _remove_angle_brackets(_extract_text(result)).replace("</s>", "").strip()
        if not text:
            text = "未生成有效提示词"
        preview = pil_to_tensor(img)
        return (text, preview)


NODE_CLASS_MAPPINGS = {
    "Shaobkj_Load_Florence2_Model": Shaobkj_Load_Florence2_Model,
    "Shaobkj_Florence2_Fast_Prompt": Shaobkj_Florence2_Fast_Prompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_Load_Florence2_Model": "🤖加载Florence2模型(急速)",
    "Shaobkj_Florence2_Fast_Prompt": "🤖图像急速反推",
}
