import base64
import io

import numpy as np

import torch
import torch.nn.functional as F
from server import PromptServer

from .shaobkj_shared import tensor_to_pil


class Shaobkj_FreeColor:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "apply_free_color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    TARGET_COLORS = (
        "全部",
        "红色",
        "黄色",
        "绿色",
        "青色",
        "蓝色",
        "洋红",
        "白色",
        "中性灰",
        "黑色",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "目标颜色": (cls.TARGET_COLORS, {"default": "全部"}),
                "色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "display": "slider"}),
                "饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "对比度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "色温": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "色调偏移": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "红通道": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "绿通道": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "蓝通道": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0, "display": "slider"}),
                "强度": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 200.0, "step": 1.0, "display": "slider"}),
            },
            "optional": {
                "遮罩": ("MASK",),
                "遮罩羽化": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1, "display": "slider"}),
                "反转遮罩": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def apply_free_color(self, **kwargs):
        image = kwargs.get("图像", kwargs.get("image"))
        target_color = kwargs.get("目标颜色", kwargs.get("target_color", "全部"))
        hue = kwargs.get("色相", kwargs.get("hue", 0.0))
        saturation = kwargs.get("饱和度", kwargs.get("saturation", 0.0))
        brightness = kwargs.get("亮度", kwargs.get("brightness", 0.0))
        contrast = kwargs.get("对比度", kwargs.get("contrast", 0.0))
        temperature = kwargs.get("色温", kwargs.get("temperature", 0.0))
        tint = kwargs.get("色调偏移", kwargs.get("tint", 0.0))
        red = kwargs.get("红通道", kwargs.get("red", 0.0))
        green = kwargs.get("绿通道", kwargs.get("green", 0.0))
        blue = kwargs.get("蓝通道", kwargs.get("blue", 0.0))
        strength = kwargs.get("强度", kwargs.get("strength", 100.0))
        mask = kwargs.get("遮罩", kwargs.get("mask"))
        mask_blur = kwargs.get("遮罩羽化", kwargs.get("mask_blur", 0.0))
        invert_mask = kwargs.get("反转遮罩", kwargs.get("invert_mask", False))
        unique_id = kwargs.get("unique_id")

        if image is None:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        batch = image.shape[0]
        outputs = []
        preview_mask = None
        for i in range(batch):
            orig = image[i]
            processed = self._process_single(
                orig,
                target_color,
                hue,
                saturation,
                brightness,
                contrast,
                temperature,
                tint,
                red,
                green,
                blue,
                strength,
            )
            if mask is not None:
                mask_tensor = self._prepare_mask(mask, i, orig.shape[0], orig.shape[1], orig.device, mask_blur, invert_mask)
                processed = orig * (1.0 - mask_tensor) + processed * mask_tensor
                if i == 0:
                    preview_mask = mask_tensor.detach().cpu()
            outputs.append(processed)
        result = torch.stack(outputs, dim=0)
        source_preview = image[0].unsqueeze(0).detach().cpu()
        output_preview = result[0].unsqueeze(0).detach().cpu()
        self._send_preview(source_preview, output_preview, unique_id, preview_mask)
        return (result,)

    def _process_single(
        self,
        image,
        target_color,
        hue,
        saturation,
        brightness,
        contrast,
        temperature,
        tint,
        red,
        green,
        blue,
        strength,
    ):
        src = torch.clamp(image, 0.0, 1.0)
        color_mask = self._build_target_mask(src, target_color).unsqueeze(-1)
        hsv = self._rgb_to_hsv(src)
        hue_shift = hue / 360.0
        sat_scale = 1.0 + saturation / 100.0
        hsv[..., 0] = torch.remainder(hsv[..., 0] + hue_shift, 1.0)
        hsv[..., 1] = torch.clamp(hsv[..., 1] * sat_scale, 0.0, 1.0)
        out = self._hsv_to_rgb(hsv)
        out = out + (brightness / 100.0)
        out = (out - 0.5) * (1.0 + contrast / 100.0) + 0.5
        temp = temperature / 100.0 * 0.25
        tint_shift = tint / 100.0 * 0.2
        out_r = out[..., 0] + temp + (red / 100.0)
        out_g = out[..., 1] - tint_shift + (green / 100.0)
        out_b = out[..., 2] - temp + (blue / 100.0)
        out = torch.stack((out_r, out_g, out_b), dim=-1)
        out = torch.clamp(out, 0.0, 1.0)
        amount = strength / 100.0
        mixed = src + (out - src) * amount * color_mask
        return torch.clamp(mixed, 0.0, 1.0)

    def _smoothstep(self, edge0, edge1, value):
        t = (value - edge0) / max(edge1 - edge0, 1e-8)
        t = torch.clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _hue_weight(self, hue, center, inner_width, outer_width):
        distance = torch.abs(torch.remainder(hue - center + 0.5, 1.0) - 0.5)
        inner = torch.tensor(inner_width, device=hue.device, dtype=hue.dtype)
        outer = torch.tensor(outer_width, device=hue.device, dtype=hue.dtype)
        return 1.0 - self._smoothstep(inner, outer, distance)

    def _build_target_mask(self, image, target_color):
        if str(target_color) == "全部":
            return torch.ones(image.shape[:2], device=image.device, dtype=image.dtype)
        hsv = self._rgb_to_hsv(image)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]
        chroma_gate = self._smoothstep(0.08, 0.2, sat) * self._smoothstep(0.08, 0.18, val)
        hue_centers = {
            "红色": 0.0,
            "黄色": 1.0 / 6.0,
            "绿色": 2.0 / 6.0,
            "青色": 3.0 / 6.0,
            "蓝色": 4.0 / 6.0,
            "洋红": 5.0 / 6.0,
        }
        if target_color in hue_centers:
            return self._hue_weight(hue, hue_centers[target_color], 0.055, 0.11) * chroma_gate
        if str(target_color) == "白色":
            return (1.0 - self._smoothstep(0.08, 0.22, sat)) * self._smoothstep(0.72, 0.92, val)
        if str(target_color) == "中性灰":
            low_sat = 1.0 - self._smoothstep(0.12, 0.28, sat)
            mid_val = self._smoothstep(0.12, 0.35, val) * (1.0 - self._smoothstep(0.68, 0.88, val))
            return low_sat * mid_val
        if str(target_color) == "黑色":
            return (1.0 - self._smoothstep(0.08, 0.25, sat)) * (1.0 - self._smoothstep(0.08, 0.3, val))
        return torch.ones(image.shape[:2], device=image.device, dtype=image.dtype)

    def _prepare_mask(self, mask, batch_index, target_h, target_w, device, mask_blur, invert_mask):
        m = mask
        if isinstance(m, torch.Tensor) and m.dim() == 3:
            if m.shape[0] > batch_index:
                m = m[batch_index]
            else:
                m = m[0]
        if isinstance(m, torch.Tensor) and m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif isinstance(m, torch.Tensor) and m.dim() == 3:
            m = m.unsqueeze(0)
        else:
            m = torch.ones((1, 1, target_h, target_w), dtype=torch.float32, device=device)
        m = m.to(device=device, dtype=torch.float32)
        m = torch.clamp(m, 0.0, 1.0)
        if m.shape[-2] != target_h or m.shape[-1] != target_w:
            m = F.interpolate(m, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if mask_blur > 0:
            kernel = int(mask_blur * 2)
            if kernel % 2 == 0:
                kernel += 1
            kernel = max(1, kernel)
            padding = kernel // 2
            m = F.avg_pool2d(m, kernel_size=kernel, stride=1, padding=padding)
        if invert_mask:
            m = 1.0 - m
        m = torch.clamp(m, 0.0, 1.0)
        return m.squeeze(0).permute(1, 2, 0)

    def _rgb_to_hsv(self, rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = torch.max(rgb, dim=-1).values
        minc = torch.min(rgb, dim=-1).values
        delta = maxc - minc
        v = maxc
        s = torch.where(maxc > 0, delta / torch.clamp(maxc, min=1e-8), torch.zeros_like(maxc))
        h = torch.zeros_like(maxc)
        nz = delta > 1e-8
        rc = (maxc - r) / torch.clamp(delta, min=1e-8)
        gc = (maxc - g) / torch.clamp(delta, min=1e-8)
        bc = (maxc - b) / torch.clamp(delta, min=1e-8)
        h = torch.where(nz & (r == maxc), bc - gc, h)
        h = torch.where(nz & (g == maxc), 2.0 + rc - bc, h)
        h = torch.where(nz & (b == maxc), 4.0 + gc - rc, h)
        h = torch.remainder(h / 6.0, 1.0)
        return torch.stack((h, s, v), dim=-1)

    def _hsv_to_rgb(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h6 = h * 6.0
        i = torch.floor(h6).to(torch.int64) % 6
        f = h6 - torch.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        shape = h.shape
        r = torch.empty(shape, device=h.device, dtype=h.dtype)
        g = torch.empty(shape, device=h.device, dtype=h.dtype)
        b = torch.empty(shape, device=h.device, dtype=h.dtype)
        m0 = i == 0
        m1 = i == 1
        m2 = i == 2
        m3 = i == 3
        m4 = i == 4
        m5 = i == 5
        r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
        return torch.stack((r, g, b), dim=-1)

    def _encode_tensor_image(self, image_tensor):
        preview_pil = tensor_to_pil(image_tensor)
        buffer = io.BytesIO()
        preview_pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _encode_mask_image(self, mask_tensor):
        if mask_tensor is None:
            return None
        if isinstance(mask_tensor, torch.Tensor):
            m = mask_tensor
            if m.dim() == 3 and m.shape[-1] == 1:
                m = m[..., 0]
            if m.dim() == 3 and m.shape[0] == 1:
                m = m[0]
            mask_np = torch.clamp(m, 0.0, 1.0).numpy()
        else:
            return None
        mask_np = (mask_np * 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        from PIL import Image
        Image.fromarray(mask_np, mode="L").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _send_preview(self, source_image, output_image, unique_id, mask_tensor=None):
        if unique_id is None or source_image is None or output_image is None:
            return
        try:
            source_encoded = self._encode_tensor_image(source_image)
            output_encoded = self._encode_tensor_image(output_image)
            mask_encoded = self._encode_mask_image(mask_tensor)
            payload = {
                "node_id": str(unique_id),
                "node_type": "Shaobkj_FreeColor",
                "image": output_encoded,
                "base_image": source_encoded,
            }
            if mask_encoded:
                payload["mask_image"] = mask_encoded
            PromptServer.instance.send_sync(
                "shaobkj.free_color.preview",
                payload,
            )
        except Exception:
            return
