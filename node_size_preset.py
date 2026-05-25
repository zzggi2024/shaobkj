import math

import torch


class Shaobkj_SizePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "数值": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "尺寸输入": (["宽", "高"], {"default": "宽"}),
                "比例": ("STRING", {"default": "9:16"}),

                "整除倍数": (["2", "4", "8", "16", "32", "64"], {"default": "8"}),
                "数值切换": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("宽", "高", "latent")
    FUNCTION = "build_size"
    CATEGORY = "🤖shaobkj-APlbox"

    def _get_ratio(self, 比例):
        parts = str(比例).split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return max(1, int(parts[0].strip())), max(1, int(parts[1].strip()))
        return 9, 16

    def _align_value(self, value, divisor):
        value = max(1, int(value))
        divisor = max(1, int(divisor))
        return max(divisor, int(round(value / divisor)) * divisor)

    def build_size(self, 数值, 尺寸输入, 比例, 整除倍数, 数值切换):
        ratio_w, ratio_h = self._get_ratio(比例)
        divisor = int(整除倍数)
        base_value = max(1, int(数值))

        if 尺寸输入 == "宽":
            width = base_value
            height = max(1, math.ceil(width * ratio_h / ratio_w))
        else:
            height = base_value
            width = max(1, math.ceil(height * ratio_w / ratio_h))

        width = self._align_value(width, divisor)
        height = self._align_value(height, divisor)

        if 数值切换:
            width, height = height, width

        latent = torch.zeros((1, 4, max(1, height // 8), max(1, width // 8)), dtype=torch.float32)
        return (width, height, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "Shaobkj_SizePreset": Shaobkj_SizePreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_SizePreset": "📐 shaobkj-尺寸预设",
}

