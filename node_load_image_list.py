import os
import torch
import numpy as np
from PIL import Image, ImageOps
import re

class Shaobkj_LoadImageListFromDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False, "placeholder": "输入文件夹路径 (如 C:\\images)", "tooltip": "图片所在文件夹路径"}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "限制加载数量，0为不限制"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "tooltip": "从第几个文件开始加载"}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "每次运行都重新加载"}),
                "sort_method": (["数字顺序", "字母顺序", "修改时间"], {"default": "数字顺序", "tooltip": "文件排序方式"}),
                "include_subdirs": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭", "tooltip": "是否递归加载子文件夹图片"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "文件名")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load_images"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method="numerical", include_subdirs=True):
        if not directory or not os.path.exists(directory):
             raise ValueError(f"❌ 错误：文件夹路径不存在: {directory}")

        if not os.path.isdir(directory):
            raise ValueError(f"❌ 错误：路径不是文件夹: {directory}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".jxl"}
        file_entries = []
        
        try:
            if include_subdirs:
                for root, dirs, files in os.walk(directory):
                    dirs.sort()
                    files.sort()
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_extensions:
                            full_path = os.path.join(root, f)
                            if os.path.isfile(full_path):
                                rel_path = os.path.relpath(full_path, directory).replace("\\", "/")
                                file_entries.append((full_path, rel_path))
            else:
                for f in os.listdir(directory):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_extensions:
                        full_path = os.path.join(directory, f)
                        if os.path.isfile(full_path):
                            rel_path = os.path.relpath(full_path, directory).replace("\\", "/")
                            file_entries.append((full_path, rel_path))
        except Exception as e:
            raise ValueError(f"❌ 错误：读取文件夹失败: {e}")

        sort_mode = {
            "数字顺序": "numerical",
            "字母顺序": "alphabetical",
            "修改时间": "date",
            "numerical": "numerical",
            "alphabetical": "alphabetical",
            "date": "date",
        }.get(str(sort_method), "numerical")

        if sort_mode == "numerical":
             def natural_sort_key(s):
                 parts = re.split('([0-9]+)', s)
                 processed = []
                 for text in parts:
                     if text.isdigit():
                         processed.append((0, int(text)))
                     else:
                         cleaned = text.strip(" ()[]{}-_")
                         processed.append((1, cleaned.lower()))
                 return processed
             
             try:
                 file_entries.sort(key=lambda x: natural_sort_key(x[1]))
             except Exception as e:
                 print(f"[Shaobkj-Loader] Numerical sort failed: {e}. Fallback to default sort.")
                 file_entries.sort(key=lambda x: x[1].lower())
        elif sort_mode == "alphabetical":
             file_entries.sort(key=lambda x: x[1].lower())
        elif sort_mode == "date":
             file_entries.sort(key=lambda x: (os.path.getmtime(x[0]), x[1].lower()))
        else:
             file_entries.sort(key=lambda x: x[1].lower())
        
        if start_index > 0:
            if start_index >= len(file_entries):
                 file_entries = []
            else:
                 file_entries = file_entries[start_index:]
        
        if image_load_cap > 0:
            file_entries = file_entries[:image_load_cap]

        if not file_entries:
             raise ValueError(f"❌ 错误：文件夹为空或筛选后无有效图片: {directory}")

        print(f"[Shaobkj-Loader] Found {len(file_entries)} images in {directory}")
        
        images_out = []
        masks_out = []
        filenames_out = []
        
        for file_path, rel_path in file_entries:
            try:
                opened_img = Image.open(file_path)
                transposed_img = ImageOps.exif_transpose(opened_img)
                img = transposed_img if transposed_img is not None else opened_img
                
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255)) or img
                image_rgb = img.convert("RGB")
                image_np = np.array(image_rgb).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                
                if 'A' in img.getbands():
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
                images_out.append(image_tensor)
                masks_out.append(mask)
                
                filename_only = os.path.splitext(rel_path)[0]
                filenames_out.append(filename_only)
                
            except Exception as e:
                print(f"[Shaobkj-Loader] Error loading {file_path}: {e}")

        if not images_out:
             raise ValueError("No images loaded successfully.")

        return (images_out, masks_out, filenames_out)
