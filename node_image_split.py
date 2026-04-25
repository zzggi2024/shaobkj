import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .shaobkj_shared import reserve_output_file_path


class Shaobkj_ImageSplit:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "split_image"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("分割图像列表", "分割图像")
    OUTPUT_IS_LIST = (True, False)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入待拆分图像"}),
                "保存目录": ("STRING", {"default": "Shaobkj_ImageSplit", "multiline": False, "tooltip": "相对输出目录的子路径；也支持绝对路径"}),
                "水平张数": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1, "tooltip": "横向拆分数量"}),
                "垂直张数": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1, "tooltip": "纵向拆分数量"}),
                "移除间距边缘": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "是否裁掉每块四周的边缘像素"}),
                "移除缩边": ("INT", {"default": 15, "min": 0, "max": 4096, "step": 1, "tooltip": "每块四周裁掉的像素值"}),
                "文件名前缀": ("STRING", {"default": "分割图像_输出", "multiline": False, "tooltip": "保存文件名前缀"}),
                "保存格式": (["PNG", "JPG", "JPEG", "WEBP", "BMP", "TIFF"], {"default": "PNG", "tooltip": "输出保存格式"}),
            }
        }

    def split_image(self, 图像, 保存目录, 水平张数, 垂直张数, 移除间距边缘, 移除缩边, 文件名前缀, 保存格式):
        images = 图像
        if isinstance(images, torch.Tensor) and images.dim() == 3:
            images = images.unsqueeze(0)
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError("❌ 错误：图像输入格式无效，需为 IMAGE 张量。")

        output_root = folder_paths.get_output_directory()
        save_dir = str(保存目录).strip() if 保存目录 is not None else ""
        if not save_dir:
            save_dir = "Shaobkj_ImageSplit"
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(output_root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        cols = max(1, int(水平张数))
        rows = max(1, int(垂直张数))
        trim_px = max(0, int(移除缩边)) if bool(移除间距边缘) else 0
        extension = str(保存格式 or "PNG").strip().lower()
        if extension == "jpg":
            extension = "jpeg"
        ext = ".jpg" if extension == "jpeg" else f".{extension}"
        base_prefix = str(文件名前缀).strip() if 文件名前缀 is not None else ""
        if not base_prefix:
            base_prefix = "分割图像_输出"

        preview_entries = []
        saved_paths = []
        split_images = []
        subfolder = os.path.relpath(save_dir, output_root).replace("\\", "/") if save_dir.startswith(output_root) else ""

        for batch_index in range(images.shape[0]):
            arr = images[batch_index].detach().cpu().numpy()
            arr = np.clip(arr, 0.0, 1.0)
            if arr.ndim != 3:
                raise ValueError("❌ 错误：单张图像维度异常。")
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=2)
            image_uint8 = (arr[:, :, :3] * 255.0).astype(np.uint8)
            h, w = image_uint8.shape[:2]
            x_edges = np.linspace(0, w, cols + 1, dtype=np.int32)
            y_edges = np.linspace(0, h, rows + 1, dtype=np.int32)

            for row in range(rows):
                for col in range(cols):
                    x0 = int(x_edges[col])
                    x1 = int(x_edges[col + 1])
                    y0 = int(y_edges[row])
                    y1 = int(y_edges[row + 1])
                    tile = image_uint8[y0:y1, x0:x1]
                    if tile.size == 0:
                        continue

                    if trim_px > 0:
                        tile_h, tile_w = tile.shape[:2]
                        max_trim_x = max(0, (tile_w - 1) // 2)
                        max_trim_y = max(0, (tile_h - 1) // 2)
                        trim_x = min(trim_px, max_trim_x)
                        trim_y = min(trim_px, max_trim_y)
                        if trim_x > 0 or trim_y > 0:
                            tile = tile[trim_y:tile_h - trim_y, trim_x:tile_w - trim_x]
                            if tile.size == 0:
                                tile = image_uint8[y0:y1, x0:x1]

                    pil_img = Image.fromarray(tile, mode="RGB")
                    name = f"{base_prefix}_b{batch_index + 1:02d}_r{row + 1:02d}_c{col + 1:02d}"
                    filename, out_path = reserve_output_file_path(save_dir, name, ext)
                    save_kwargs = {}
                    if extension == "jpeg":
                        save_kwargs = {"format": "JPEG", "quality": 95}
                    elif extension == "png":
                        save_kwargs = {"format": "PNG"}
                    elif extension == "webp":
                        save_kwargs = {"format": "WEBP", "quality": 95}
                    elif extension == "bmp":
                        save_kwargs = {"format": "BMP"}
                    elif extension == "tiff":
                        save_kwargs = {"format": "TIFF"}
                    else:
                        save_kwargs = {"format": "PNG"}
                    pil_img.save(out_path, **save_kwargs)
                    split_tensor = torch.from_numpy(tile.astype(np.float32) / 255.0).unsqueeze(0)
                    split_images.append(split_tensor)
                    saved_paths.append(out_path)
                    if subfolder:
                        preview_entries.append({"filename": filename, "subfolder": subfolder, "type": "output"})

        if not saved_paths:
            raise ValueError("❌ 错误：没有成功拆分出任何图像。")

        result_dir = save_dir.replace("\\", "/")
        if preview_entries:
            return {"ui": {"images": preview_entries}, "result": (split_images, result_dir)}
        return (split_images, result_dir)
