import os
import re

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import Types


class Shaobkj_CustomVideoSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "要保存的视频输入"}),
                "保存目录": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False, "tooltip": "视频保存目录，支持绝对路径；目录不存在会自动创建"}),
                "视频名称": ("STRING", {"default": "ComfyUI", "multiline": False, "tooltip": "不需要填写扩展名；重名会自动追加 _1、_2..."}),
                "format": (Types.VideoContainer.as_input(), {"default": "auto", "tooltip": "视频封装格式"}),
                "codec": (Types.VideoCodec.as_input(), {"default": "auto", "tooltip": "视频编码"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("保存路径",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    @staticmethod
    def _safe_filename(name):
        name = str(name or "").strip()
        name = os.path.basename(name)
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
        return name or "ComfyUI"

    @staticmethod
    def _unique_path(directory, filename, extension):
        base = filename
        target = os.path.join(directory, f"{base}.{extension}")
        if not os.path.exists(target):
            return target
        index = 1
        while True:
            target = os.path.join(directory, f"{base}_{index}.{extension}")
            if not os.path.exists(target):
                return target
            index += 1

    def save_video(self, video, 保存目录, 视频名称, format, codec, prompt=None, extra_pnginfo=None):
        output_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(str(保存目录 or "").strip())))
        os.makedirs(output_dir, exist_ok=True)

        extension = Types.VideoContainer.get_extension(format) or "mp4"
        filename = self._safe_filename(视频名称)
        if filename.lower().endswith(f".{extension.lower()}"):
            filename = filename[:-(len(extension) + 1)].strip(" .") or "ComfyUI"

        output_path = self._unique_path(output_dir, filename, extension)

        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if metadata:
                saved_metadata = metadata

        video.save_to(
            output_path,
            format=Types.VideoContainer(format),
            codec=codec,
            metadata=saved_metadata,
        )
        return (output_path,)
