import os
import re
import json
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
    "描述": "<DESCRIPTION>",
    "generate tags(PromptGen 1.5)": "<GENERATE_TAGS>",
    "mixed caption(PromptGen 1.5)": "<MIXED_CAPTION>",
    "mixed caption plus(PromptGen 2.0)": "<MIXED_CAPTION_PLUS>",
    "analyze(PromptGen 2.0)": "<<ANALYZE>>",
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
        # 清理多余的列表格式（例如 "['...']"）
        text = result.strip()
        if text.startswith("['") and text.endswith("']"):
            text = text[2:-2]
        elif text.startswith('["') and text.endswith('"]'):
            text = text[2:-2]
        return text
    if isinstance(result, dict):
        if "labels" in result and isinstance(result["labels"], list):
            labels = [str(x).replace("</s>", "").replace("<s>", "").strip() for x in result["labels"] if str(x).strip()]
            if labels:
                return ", ".join(labels)
        # 尝试提取第一个有效的值
        for key, value in result.items():
            text = _extract_text(value)
            if text:
                return text
    if isinstance(result, list):
        texts = []
        for item in result:
            t = _extract_text(item)
            if t:
                texts.append(t)
        return ", ".join(texts)
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

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            changed = False
            if isinstance(cfg, dict):
                text_cfg = cfg.get("text_config")
                if isinstance(text_cfg, dict):
                    if "forced_bos_token_id" not in text_cfg:
                        text_cfg["forced_bos_token_id"] = text_cfg.get("bos_token_id")
                        changed = True
                    if "forced_eos_token_id" not in text_cfg and "eos_token_id" in text_cfg:
                        text_cfg["forced_eos_token_id"] = text_cfg.get("eos_token_id")
                        changed = True
                if "forced_bos_token_id" not in cfg and isinstance(text_cfg, dict):
                    cfg["forced_bos_token_id"] = text_cfg.get("forced_bos_token_id")
                    changed = True
                if "forced_eos_token_id" not in cfg and isinstance(text_cfg, dict) and "forced_eos_token_id" in text_cfg:
                    cfg["forced_eos_token_id"] = text_cfg.get("forced_eos_token_id")
                    changed = True
            if changed:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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
                
                # Dynamically patch GenerationMixin if it's missing (fixes transformers >= 4.45 compatibility)
                if hasattr(model, "language_model") and not hasattr(model.language_model, "generate"):
                    try:
                        from transformers import GenerationMixin
                        if GenerationMixin not in model.language_model.__class__.__bases__:
                            model.language_model.__class__.__bases__ = (GenerationMixin,) + model.language_model.__class__.__bases__
                    except ImportError:
                        pass
                
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model = model.to(device)
                if not hasattr(model.config, "forced_bos_token_id"):
                    setattr(model.config, "forced_bos_token_id", getattr(model.config, "bos_token_id", None))
                if not hasattr(model.config, "forced_eos_token_id"):
                    setattr(model.config, "forced_eos_token_id", getattr(model.config, "eos_token_id", None))
                if hasattr(model.config, "text_config") and model.config.text_config is not None:
                    if not hasattr(model.config.text_config, "forced_bos_token_id"):
                        setattr(
                            model.config.text_config,
                            "forced_bos_token_id",
                            getattr(model.config.text_config, "bos_token_id", None),
                        )
                    if not hasattr(model.config.text_config, "forced_eos_token_id"):
                        setattr(
                            model.config.text_config,
                            "forced_eos_token_id",
                            getattr(model.config.text_config, "eos_token_id", None),
                        )
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


def _run_florence(model_bundle, image, task, text_input, max_new_tokens, num_beams, seed):
    if not isinstance(model_bundle, dict):
        raise RuntimeError("Florence2模型对象无效，请重新连接‘🤖加载Florence2模型(急速)’输出。")

    task_token = TASK_TOKEN_MAP.get(task, "<MORE_DETAILED_CAPTION>")
    
    # 修复：Florence-2 的 processor 需要独立的 text 仅包含 task_token，
    # 若有额外输入，应在 task_token 之后拼接，并保持内部处理逻辑兼容
    prompt = task_token if not text_input else f"{task_token}{text_input}"
    
    # 特殊处理：部分 transformers 版本的 processor 内部断言要求传入的 text 完全等于 task_token
    # 如果用户有额外输入，我们必须绕过它，或者直接将 task_token 传给 text，后续文本在处理时可能会丢失。
    # 实际上，大多数官方文档要求：对于带输入的任务，text = task_token + text_input。
    # 如果引发了 AssertionError: Task token ... should be the only token in the text.
    # 这是因为该任务不支持附加文本输入，或者 processor 强校验了该任务类型。
    # 我们这里在传给 processor 时，仍然传递完整 prompt，如果遇到强校验，我们捕获并退回到只传 task_token。
    
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

    # Set seed
    import random
    from transformers import set_seed
    if seed is not None:
        safe_seed = seed
        if safe_seed < 0:
            safe_seed = random.randint(0, 2147483647)
        if safe_seed > 2147483647:
            safe_seed = safe_seed % 2147483647
        torch.manual_seed(safe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(safe_seed)
        set_seed(safe_seed)

    # 强制采样：只要种子大于 0（非 0），我们就自动开启采样，以使随机种子能够真正影响文本生成。
    # 如果种子为 0（固定默认），则不使用采样，走贪婪搜索，保证结果绝对一致且无随机性。
    is_sample = False
    temperature = None
    final_num_beams = int(num_beams)
    if seed is not None and seed > 0:
        is_sample = True
        temperature = 0.7
        final_num_beams = 1  # 开启采样时，强制束搜索数为 1 才能最大化随机性

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except AssertionError as e:
        if "should be the only token" in str(e):
            print(f"[ComfyUI-shaobkj] 警告: 任务 '{task}' 不支持附加文本输入，已自动忽略额外文本。")
            inputs = processor(text=task_token, images=image, return_tensors="pt")
        else:
            raise e
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    from comfy.utils import ProgressBar
    import threading
    import time
    
    pbar = ProgressBar(100)
    pbar.update_absolute(0)
    
    progress_state = {"stop": False}

    def progress_simulator():
        current = 0.0
        while not progress_state["stop"] and current < 95.0:
            time.sleep(0.1)
            if current < 30: 
                step = 5.0
            elif current < 60: 
                step = 2.0
            elif current < 80: 
                step = 1.0
            else: 
                step = 0.2
            current += step
            if current > 95: current = 95
            pbar.update_absolute(int(current))

    t_progress = threading.Thread(target=progress_simulator)
    t_progress.daemon = True
    t_progress.start()

    try:
        with torch.inference_mode():
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "pixel_values": inputs["pixel_values"],
                "max_new_tokens": int(max_new_tokens),
                "early_stopping": False,
                "do_sample": is_sample,
                "num_beams": final_num_beams,
                "use_cache": True,
            }
            if is_sample and temperature is not None:
                gen_kwargs["temperature"] = temperature
                
            if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type="cuda", dtype=dtype):
                    generated_ids = model.generate(**gen_kwargs)
            else:
                generated_ids = model.generate(**gen_kwargs)
    finally:
        progress_state["stop"] = True
        pbar.update_absolute(100)

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "tooltip": "固定随机种子；推荐：0"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("提示词", "预览图")
    FUNCTION = "run"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    @classmethod
    def IS_CHANGED(cls, Florence2模型, 图像, 任务, 文本输入, 最大新token, 束搜索数, seed):
        return seed

    def run(self, Florence2模型, 图像, 任务, 文本输入, 最大新token, 束搜索数, seed):
        img = tensor_to_pil(图像[0]).convert("RGB")
        result = _run_florence(
            Florence2模型,
            img,
            任务,
            文本输入,
            最大新token,
            束搜索数,
            seed,
        )
        text = _remove_angle_brackets(_extract_text(result)).replace("</s>", "").replace("<s>", "").strip()
        
        # 移除可能残留的任务名称（例如从字典中提取时带出的键名，或生成的特殊前缀）
        for task_key in TASK_TOKEN_MAP.keys():
            if text.startswith(f"{task_key}:"):
                text = text[len(task_key)+1:].strip()
            if text.startswith(f"'{task_key}':"):
                text = text[len(task_key)+3:].strip()
            if text.startswith(f'"{task_key}":'):
                text = text[len(task_key)+3:].strip()
        
        # 移除可能存在的被花括号或方括号包裹的情况（例如 "{'caption': '...'}"）
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1].strip()
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1].strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

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
