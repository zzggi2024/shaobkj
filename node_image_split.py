import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .shaobkj_shared import reserve_output_file_path


def _find_separator_interval(profile, target, image_length, split_count):
    if split_count <= 1:
        return None
    search_radius = max(8, image_length // max(8, split_count * 4))
    left = max(1, int(target) - search_radius)
    right = min(image_length - 1, int(target) + search_radius)
    if right <= left:
        return None

    candidates = np.where(profile[left:right] >= 0.82)[0]
    if candidates.size == 0:
        return None
    candidates = candidates + left

    runs = []
    start = int(candidates[0])
    prev = int(candidates[0])
    for value in candidates[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev + 1))
            start = value
            prev = value
    runs.append((start, prev + 1))

    best = min(runs, key=lambda item: abs(((item[0] + item[1]) / 2) - target))
    if best[1] - best[0] < 2 and image_length > 256:
        return None
    return best


def _build_split_intervals(image_uint8, count, axis, mode):
    length = image_uint8.shape[1] if axis == 1 else image_uint8.shape[0]
    count = max(1, int(count))
    if count <= 1:
        return [(0, length)]

    if mode != "智能分隔线":
        return [(i * length // count, (i + 1) * length // count) for i in range(count)]

    white_mask = np.all(image_uint8 >= 235, axis=2)
    profile = white_mask.mean(axis=0 if axis == 1 else 1)
    separators = []
    for i in range(1, count):
        target = i * length / count
        interval = _find_separator_interval(profile, target, length, count)
        if interval is None:
            return [(j * length // count, (j + 1) * length // count) for j in range(count)]
        separators.append(interval)

    intervals = []
    start = 0
    for sep_start, sep_end in separators:
        intervals.append((start, sep_start))
        start = sep_end
    intervals.append((start, length))

    if len(intervals) != count or any(end <= start for start, end in intervals):
        return [(i * length // count, (i + 1) * length // count) for i in range(count)]
    return intervals


class Shaobkj_ImageSplit:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "split_image"
    MAX_SPLIT_OUTPUTS = 64
    RETURN_TYPES = tuple("IMAGE" for _ in range(MAX_SPLIT_OUTPUTS))
    RETURN_NAMES = tuple(f"裁切图像{i}" for i in range(1, MAX_SPLIT_OUTPUTS + 1))
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入待拆分图像"}),
                "保存目录": ("STRING", {"default": "Shaobkj_ImageSplit", "multiline": False, "tooltip": "相对输出目录的子路径；也支持绝对路径"}),
                "水平张数": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1, "tooltip": "横向拆分数量，最多 8 张"}),
                "垂直张数": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1, "tooltip": "纵向拆分数量，最多 8 张"}),
                "裁切模式": (["等分网格", "智能分隔线"], {"default": "智能分隔线", "tooltip": "智能分隔线会自动识别白色分隔带，识别失败则回退为等分网格"}),
                "移除间距边缘": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "是否裁掉每块四周的边缘像素"}),
                "移除缩边": ("INT", {"default": 15, "min": 0, "max": 4096, "step": 1, "tooltip": "每块四周裁掉的像素值"}),
                "文件名前缀": ("STRING", {"default": "分割图像_输出", "multiline": False, "tooltip": "保存文件名前缀"}),
                "保存格式": (["PNG", "JPG", "JPEG", "WEBP", "BMP", "TIFF"], {"default": "PNG", "tooltip": "输出保存格式"}),
            }
        }

    def split_image(self, 图像, 保存目录, 水平张数, 垂直张数, 裁切模式, 移除间距边缘, 移除缩边, 文件名前缀, 保存格式):
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

        cols = min(8, max(1, int(水平张数)))
        rows = min(8, max(1, int(垂直张数)))
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
            x_intervals = _build_split_intervals(image_uint8, cols, 1, 裁切模式)
            y_intervals = _build_split_intervals(image_uint8, rows, 0, 裁切模式)

            for row in range(rows):
                for col in range(cols):
                    x0, x1 = x_intervals[col]
                    y0, y1 = y_intervals[row]
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

        result = tuple(split_images[i] if i < len(split_images) else split_images[-1] for i in range(self.MAX_SPLIT_OUTPUTS))
        if preview_entries:
            return {"ui": {"images": preview_entries}, "result": result}
        return result
