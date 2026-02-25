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
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "每次运行都重新加载"}),
                "sort_method": (["numerical", "alphabetical", "date"], {"default": "numerical", "tooltip": "文件排序方式"}),
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

    def load_images(self, directory: str, image_load_cap: int = 0, start_index: int = 0, load_always=False, sort_method="numerical"):
        if not directory or not os.path.exists(directory):
             raise ValueError(f"❌ 错误：文件夹路径不存在: {directory}")

        if not os.path.isdir(directory):
            raise ValueError(f"❌ 错误：路径不是文件夹: {directory}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".jxl"}
        file_list = []
        
        try:
            for f in os.listdir(directory):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(directory, f)
                    if os.path.isfile(full_path):
                        file_list.append(full_path)
        except Exception as e:
            raise ValueError(f"❌ 错误：读取文件夹失败: {e}")

        if sort_method == "numerical":
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
                 file_list.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
             except Exception as e:
                 print(f"[Shaobkj-Loader] Numerical sort failed: {e}. Fallback to default sort.")
                 file_list.sort()
        elif sort_method == "alphabetical":
             file_list.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_method == "date":
             file_list.sort(key=lambda x: os.path.getmtime(x))
        else:
             file_list.sort()
        
        if start_index > 0:
            if start_index >= len(file_list):
                 file_list = []
            else:
                 file_list = file_list[start_index:]
        
        if image_load_cap > 0:
            file_list = file_list[:image_load_cap]

        if not file_list:
             raise ValueError(f"❌ 错误：文件夹为空或筛选后无有效图片: {directory}")

        print(f"[Shaobkj-Loader] Found {len(file_list)} images in {directory}")
        
        images_out = []
        masks_out = []
        filenames_out = []
        
        for file_path in file_list:
            try:
                img = Image.open(file_path)
                img = ImageOps.exif_transpose(img)
                
                if img.mode == 'I':
                    img = img.point(lambda i: i * (1 / 255))
                image = img.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                
                if 'A' in img.getbands():
                    mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
                images_out.append(image)
                masks_out.append(mask)
                
                # Extract filename without extension and path
                filename_only = os.path.splitext(os.path.basename(file_path))[0]
                filenames_out.append(filename_only)
                
            except Exception as e:
                print(f"[Shaobkj-Loader] Error loading {file_path}: {e}")

        if not images_out:
             raise ValueError("No images loaded successfully.")

        return (images_out, masks_out, filenames_out)
