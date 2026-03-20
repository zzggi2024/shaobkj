"""
接回头节点 - 四方连续无缝拼接
Seamless Pattern Node - Create tileable patterns for textile printing

支持多种接法：
- 跳接 (1/2): 纵向错位 1/2 重复排列
- 平接一：上下左右规律性重复排列
- 上下连接：二方连续，上下重复连续排列
- 左右连接：二方连续，左右重复连续排列
- 平接二：另一种平接模式
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import os


class Shaobkj_SeamlessPattern:
    """
    接回头节点 - 将普通图片转换为四方连续无缝拼接图
    """
    
    CATEGORY = "shaobkj/image"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("seamless_image",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pattern_type": (
                    [
                        "jump_half",      # 跳接 (1/2)
                        "plain_1",        # 平接一
                        "vertical_2way",  # 上下连接
                        "horizontal_2way",# 左右连接
                        "plain_2",        # 平接二
                    ],
                    {
                        "default": "jump_half",
                        "labels": {
                            "jump_half": "跳接 (1/2) - 纵向错位 1/2 重复",
                            "plain_1": "平接一 - 上下左右规律重复",
                            "vertical_2way": "上下连接 - 二方连续",
                            "horizontal_2way": "左右连接 - 二方连续",
                            "plain_2": "平接二 - 另一种平接",
                        }
                    }
                ),
                "blend_width": ("INT", {
                    "default": 150,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "display": "number"
                }),
                "output_scale": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "number",
                    "tooltip": "输出图像的行列数 (2=2x2, 3=3x3, 4=4x4)"
                }),
            }
        }
    
    def execute(self, image, pattern_type, blend_width, output_scale):
        """
        执行接回头处理
        
        Args:
            image: 输入图像张量 [B, H, W, C]
            pattern_type: 接法类型
            blend_width: 边缘融合宽度 (像素)
            output_scale: 输出缩放倍数 (2=2x2, 3=3x3)
        
        Returns:
            四方连续图像张量
        """
        # 获取第一张图像 (batch 中的第一个)
        img_tensor = image[0]  # [H, W, C]
        
        # 转换为 numpy 数组 (0-255)
        img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        
        h, w = img_np.shape[:2]
        
        # 1. 边缘修复 - 让上下左右可以无缝对接
        seamless_img = self._fix_edges(img_np, blend_width)
        
        # 2. 按选定方式拼接
        if pattern_type == "jump_half":
            result = self._jump_half_pattern(seamless_img, blend_width, output_scale)
        elif pattern_type == "plain_1":
            result = self._plain_pattern(seamless_img, blend_width, output_scale, offset=False)
        elif pattern_type == "vertical_2way":
            result = self._vertical_pattern(seamless_img, blend_width, output_scale)
        elif pattern_type == "horizontal_2way":
            result = self._horizontal_pattern(seamless_img, blend_width, output_scale)
        elif pattern_type == "plain_2":
            result = self._plain_pattern(seamless_img, blend_width, output_scale, offset=True)
        else:
            result = seamless_img
        
        # 3. 转换回张量格式
        result_tensor = torch.from_numpy(result).float() / 255.0
        result_tensor = result_tensor.unsqueeze(0)  # [1, H, W, C]
        
        return (result_tensor,)
    
    def _fix_edges(self, img, blend_width):
        """
        边缘修复 - 使用渐变融合让边缘可以无缝对接
        
        这个方法使用简单的渐变混合，更高级的版本可以使用 AI Inpaint
        """
        h, w = img.shape[:2]
        result = img.copy()
        
        # 创建融合蒙版
        # 上边缘：从 0 到 blend_width，透明度从 0 到 1
        top_mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
        top_mask = np.tile(top_mask, (1, w))
        
        # 下边缘：从 h-blend_width 到 h，透明度从 1 到 0
        bottom_mask = np.linspace(1, 0, blend_width).reshape(-1, 1)
        bottom_mask = np.tile(bottom_mask, (1, w))
        
        # 左边缘
        left_mask = np.linspace(0, 1, blend_width).reshape(1, -1)
        left_mask = np.tile(left_mask, (h, 1))
        
        # 右边缘
        right_mask = np.linspace(1, 0, blend_width).reshape(1, -1)
        right_mask = np.tile(right_mask, (h, 1))
        
        # 处理上下边缘融合
        if blend_width < h // 2:
            # 将底部区域混合到顶部
            top_region = result[0:blend_width, :]
            bottom_region = result[h-blend_width:h, :]
            
            # 交叉融合
            blended_top = (top_region * (1 - top_mask[:, :, None]) + bottom_region * top_mask[:, :, None]).astype(np.uint8)
            blended_bottom = (bottom_region * (1 - bottom_mask[:, :, None]) + top_region * bottom_mask[:, :, None]).astype(np.uint8)
            
            result[0:blend_width, :] = blended_top
            result[h-blend_width:h, :] = blended_bottom
        
        # 处理左右边缘融合
        if blend_width < w // 2:
            # 将右区域混合到左区域
            left_region = result[:, 0:blend_width]
            right_region = result[:, w-blend_width:w]
            
            # 交叉融合
            blended_left = (left_region * (1 - left_mask[:, :, None]) + right_region * left_mask[:, :, None]).astype(np.uint8)
            blended_right = (right_region * (1 - right_mask[:, :, None]) + left_region * right_mask[:, :, None]).astype(np.uint8)
            
            result[:, 0:blend_width] = blended_left
            result[:, w-blend_width:w] = blended_right
        
        return result
    
    def _jump_half_pattern(self, img, blend_width, scale):
        """
        跳接 (1/2) - 每行向右偏移 50%
        
        排列方式：
        Row 0: [A][A][A]...
        Row 1:   [A][A][A]...  (偏移 50%)
        Row 2: [A][A][A]...
        """
        h, w = img.shape[:2]
        offset = w // 2
        
        # 计算输出尺寸
        out_h = h * scale
        out_w = w * scale
        
        # 创建输出画布
        if len(img.shape) == 3:
            result = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        # 填充每一行
        for row in range(scale):
            for col in range(scale):
                # 计算起始位置
                y_start = row * h
                x_start = col * w
                
                # 奇数行向右偏移 50%
                if row % 2 == 1:
                    x_start += offset
                
                # 处理图像 wrapping
                self._place_image_wrapped(result, img, y_start, x_start, blend_width)
        
        return result
    
    def _plain_pattern(self, img, blend_width, scale, offset=False):
        """
        平接 - 直接网格排列
        
        offset=True 时为平接二（偶数行偏移）
        """
        h, w = img.shape[:2]
        
        out_h = h * scale
        out_w = w * scale
        
        if len(img.shape) == 3:
            result = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        for row in range(scale):
            for col in range(scale):
                y_start = row * h
                x_start = col * w
                
                # 平接二：偶数行偏移
                if offset and row % 2 == 0:
                    x_start += w // 2
                
                self._place_image_wrapped(result, img, y_start, x_start, blend_width)
        
        return result
    
    def _vertical_pattern(self, img, blend_width, scale):
        """
        上下连接 - 仅纵向重复 (二方连续)
        """
        h, w = img.shape[:2]
        
        out_h = h * scale
        out_w = w
        
        if len(img.shape) == 3:
            result = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        for row in range(scale):
            y_start = row * h
            result[y_start:y_start+h, :] = img
        
        return result
    
    def _horizontal_pattern(self, img, blend_width, scale):
        """
        左右连接 - 仅横向重复 (二方连续)
        """
        h, w = img.shape[:2]
        
        out_h = h
        out_w = w * scale
        
        if len(img.shape) == 3:
            result = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((out_h, out_w), dtype=np.uint8)
        
        for col in range(scale):
            x_start = col * w
            result[:, x_start:x_start+w] = img
        
        return result
    
    def _place_image_wrapped(self, canvas, img, y_start, x_start, blend_width):
        """
        将图像放置到画布上，处理边界 wrapping
        
        如果图像超出画布边界，会自动 wrapping 到另一侧
        """
        h, w = img.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        
        # 处理负 x 起始位置 (向左偏移时)
        if x_start < 0:
            # 右侧部分
            right_part = img[:, :w+x_start]
            self._place_image_wrapped(canvas, right_part, y_start, canvas_w + x_start, blend_width)
            # 左侧部分
            left_part = img[:, w+x_start:]
            self._place_image_wrapped(canvas, left_part, y_start, 0, blend_width)
            return
        
        # 处理负 y 起始位置
        if y_start < 0:
            # 下侧部分
            bottom_part = img[:h+y_start, :]
            self._place_image_wrapped(canvas, bottom_part, canvas_h + y_start, x_start, blend_width)
            # 上侧部分
            top_part = img[h+y_start:, :]
            self._place_image_wrapped(canvas, top_part, 0, x_start, blend_width)
            return
        
        # 处理超出右边界
        if x_start + w > canvas_w:
            # 左侧部分
            left_part = img[:, :canvas_w-x_start]
            self._place_image_wrapped(canvas, left_part, y_start, x_start, blend_width)
            # 右侧部分 (wrapping 到左边)
            right_part = img[:, canvas_w-x_start:]
            self._place_image_wrapped(canvas, right_part, y_start, 0, blend_width)
            return
        
        # 处理超出下边界
        if y_start + h > canvas_h:
            # 上侧部分
            top_part = img[:canvas_h-y_start, :]
            self._place_image_wrapped(canvas, top_part, y_start, x_start, blend_width)
            # 下侧部分 (wrapping 到上边)
            bottom_part = img[canvas_h-y_start:, :]
            self._place_image_wrapped(canvas, bottom_part, 0, x_start, blend_width)
            return
        
        # 正常放置
        if 0 <= y_start < canvas_h and 0 <= x_start < canvas_w:
            y_end = min(y_start + h, canvas_h)
            x_end = min(x_start + w, canvas_w)
            
            img_h = y_end - y_start
            img_w = x_end - x_start
            
            if img_h > 0 and img_w > 0:
                canvas[y_start:y_end, x_start:x_end] = img[0:img_h, 0:img_w]


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Shaobkj_SeamlessPattern": Shaobkj_SeamlessPattern,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_SeamlessPattern": "🎨 接回头 (四方连续)",
}
