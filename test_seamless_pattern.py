"""
测试接回头节点功能
"""

import sys
import os
sys.path.insert(0, r"D:\ComfyUI-GPU-Shaobkj\ComfyUI")

import torch
import numpy as np
from node_seamless_pattern import Shaobkj_SeamlessPattern

def test_seamless_pattern():
    """测试接回头节点"""
    
    # 创建一个测试图像 (渐变图案)
    h, w = 256, 256
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 创建对角线渐变
    for y in range(h):
        for x in range(w):
            test_img[y, x, 0] = int(255 * x / w)  # R
            test_img[y, x, 1] = int(255 * y / h)  # G
            test_img[y, x, 2] = 128               # B
    
    # 转换为张量格式 [B, H, W, C]
    img_tensor = torch.from_numpy(test_img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    print(f"输入图像尺寸：{img_tensor.shape}")
    
    # 创建节点实例
    node = Shaobkj_SeamlessPattern()
    
    # 测试跳接 (1/2)
    print("\n=== 测试跳接 (1/2) ===")
    result = node.execute(img_tensor, "jump_half", 50, 2)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h*2, w*2, 3), f"期望 (1, {h*2}, {w*2}, 3), 得到 {result[0].shape}"
    
    # 测试平接一
    print("\n=== 测试平接一 ===")
    result = node.execute(img_tensor, "plain_1", 50, 2)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h*2, w*2, 3)
    
    # 测试上下连接
    print("\n=== 测试上下连接 ===")
    result = node.execute(img_tensor, "vertical_2way", 50, 2)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h*2, w, 3)
    
    # 测试左右连接
    print("\n=== 测试左右连接 ===")
    result = node.execute(img_tensor, "horizontal_2way", 50, 2)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h, w*2, 3)
    
    # 测试平接二
    print("\n=== 测试平接二 ===")
    result = node.execute(img_tensor, "plain_2", 50, 2)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h*2, w*2, 3)
    
    # 测试 3x3 输出
    print("\n=== 测试 3x3 输出 ===")
    result = node.execute(img_tensor, "jump_half", 50, 3)
    print(f"输出图像尺寸：{result[0].shape}")
    assert result[0].shape == (1, h*3, w*3, 3)
    
    print("\n[OK] All tests passed!")
    
    # 保存测试图像
    output_img = result[0][0].cpu().numpy()
    output_img = np.clip(255.0 * output_img, 0, 255).astype(np.uint8)
    
    from PIL import Image
    output_path = r"D:\ComfyUI-GPU-Shaobkj\ComfyUI\output\test_seamless_pattern.png"
    Image.fromarray(output_img).save(output_path)
    print(f"\n[IMG] Test image saved to: {output_path}")

if __name__ == "__main__":
    test_seamless_pattern()
