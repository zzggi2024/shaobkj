import os
import re
import shutil

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

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "🤖shaobkj-APlbox/实用工具"

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

    @staticmethod
    def _preview_entry(output_path):
        output_dir = os.path.abspath(folder_paths.get_output_directory())
        temp_dir = os.path.abspath(folder_paths.get_temp_directory())
        abs_path = os.path.abspath(output_path)
        filename = os.path.basename(abs_path)

        try:
            rel = os.path.relpath(abs_path, output_dir)
            if not rel.startswith("..") and not os.path.isabs(rel):
                subfolder = os.path.dirname(rel).replace("\\", "/")
                return {"filename": filename, "subfolder": subfolder, "type": "output"}
        except ValueError:
            pass

        preview_name = filename
        preview_path = os.path.join(temp_dir, preview_name)
        if os.path.abspath(preview_path) != abs_path:
            preview_path = Shaobkj_CustomVideoSave._unique_path(temp_dir, os.path.splitext(filename)[0], os.path.splitext(filename)[1].lstrip(".") or "mp4")
            shutil.copyfile(abs_path, preview_path)
            preview_name = os.path.basename(preview_path)
        return {"filename": preview_name, "subfolder": "", "type": "temp"}

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
        return {"ui": {"images": [self._preview_entry(output_path)], "animated": (True,)}}
