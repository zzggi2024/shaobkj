import base64
import io
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .shaobkj_shared import pil_to_tensor


class Shaobkj_QuickMark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brush_data": ("STRING", {"default": "", "multiline": True}),
                "brush_size": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "image_base64": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("原图", "合成图", "总mask", "黑mask", "白mask", "红mask", "绿mask", "蓝mask", "灰mask")
    FUNCTION = "main"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def main(self, brush_data, brush_size, image_base64):
        brush_data = self._normalize_text_input(brush_data)
        image_base64 = self._normalize_text_input(image_base64)

        background_img_tensor = self._decode_background_image(image_base64)
        if background_img_tensor is None:
            background_img_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        batch_size, height, width = background_img_tensor.shape[:3]

        black_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        white_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        red_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        green_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        blue_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        gray_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        marker_annotations = []

        if brush_data.strip():
            self._apply_brush_data(
                brush_data=brush_data,
                brush_size=brush_size,
                width=width,
                height=height,
                black_mask=black_mask,
                white_mask=white_mask,
                red_mask=red_mask,
                green_mask=green_mask,
                blue_mask=blue_mask,
                gray_mask=gray_mask,
                marker_annotations=marker_annotations,
            )

        black_mask = torch.clamp(black_mask, 0.0, 1.0)
        white_mask = torch.clamp(white_mask, 0.0, 1.0)
        red_mask = torch.clamp(red_mask, 0.0, 1.0)
        green_mask = torch.clamp(green_mask, 0.0, 1.0)
        blue_mask = torch.clamp(blue_mask, 0.0, 1.0)
        gray_mask = torch.clamp(gray_mask, 0.0, 1.0)

        sum_mask = torch.maximum(black_mask, white_mask)
        sum_mask = torch.maximum(sum_mask, red_mask)
        sum_mask = torch.maximum(sum_mask, green_mask)
        sum_mask = torch.maximum(sum_mask, blue_mask)
        sum_mask = torch.maximum(sum_mask, gray_mask)

        sum_image = background_img_tensor.clone()

        black_mask_4d = black_mask.unsqueeze(-1)
        white_mask_4d = white_mask.unsqueeze(-1)
        red_mask_4d = red_mask.unsqueeze(-1)
        green_mask_4d = green_mask.unsqueeze(-1)
        blue_mask_4d = blue_mask.unsqueeze(-1)
        gray_mask_4d = gray_mask.unsqueeze(-1)

        sum_image = sum_image * (1 - black_mask_4d) + torch.tensor([0.0, 0.0, 0.0], dtype=sum_image.dtype, device=sum_image.device) * black_mask_4d
        sum_image = sum_image * (1 - white_mask_4d) + torch.tensor([1.0, 1.0, 1.0], dtype=sum_image.dtype, device=sum_image.device) * white_mask_4d
        sum_image = sum_image * (1 - red_mask_4d) + torch.tensor([1.0, 0.0, 0.0], dtype=sum_image.dtype, device=sum_image.device) * red_mask_4d
        sum_image = sum_image * (1 - green_mask_4d) + torch.tensor([0.0, 1.0, 0.0], dtype=sum_image.dtype, device=sum_image.device) * green_mask_4d
        sum_image = sum_image * (1 - blue_mask_4d) + torch.tensor([0.0, 0.0, 1.0], dtype=sum_image.dtype, device=sum_image.device) * blue_mask_4d
        sum_image = sum_image * (1 - gray_mask_4d) + torch.tensor([0.5, 0.5, 0.5], dtype=sum_image.dtype, device=sum_image.device) * gray_mask_4d

        if marker_annotations:
            sum_image = self._draw_marker_annotations(sum_image, marker_annotations)

        return (background_img_tensor, sum_image, sum_mask, black_mask, white_mask, red_mask, green_mask, blue_mask, gray_mask)

    def _normalize_text_input(self, value):
        if isinstance(value, (list, tuple)):
            return next((item for item in value if isinstance(item, str) and item.strip()), "") if value else ""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _decode_background_image(self, image_base64):
        if not image_base64.strip():
            return None
        try:
            base64_data = image_base64.strip()
            if "," in base64_data:
                base64_data = base64_data.split(",")[-1]
            image_bytes = base64.b64decode(base64_data)
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return pil_to_tensor(image_pil)
        except Exception as error:
            print(f"[Shaobkj_QuickMark] Failed to decode image_base64: {error}")
            return None

    def _apply_brush_data(
        self,
        brush_data,
        brush_size,
        width,
        height,
        black_mask,
        white_mask,
        red_mask,
        green_mask,
        blue_mask,
        gray_mask,
        marker_annotations,
    ):
        color_mapping = {
            "0,0,0": "black",
            "255,255,255": "white",
            "255,0,0": "red",
            "0,255,0": "green",
            "0,0,255": "blue",
            "128,128,128": "gray",
        }

        try:
            strokes = brush_data.split("|")

            black_mask_np = black_mask[0].cpu().numpy().copy()
            white_mask_np = white_mask[0].cpu().numpy().copy()
            red_mask_np = red_mask[0].cpu().numpy().copy()
            green_mask_np = green_mask[0].cpu().numpy().copy()
            blue_mask_np = blue_mask[0].cpu().numpy().copy()
            gray_mask_np = gray_mask[0].cpu().numpy().copy()

            for stroke in strokes:
                if not stroke.strip():
                    continue

                mode = "brush"
                stroke_type = "free"
                stroke_size = brush_size
                stroke_opacity = 1.0
                stroke_color = "255,255,255"
                points_str = stroke
                marker_id = None

                if ":" in stroke:
                    parts = stroke.split(":")
                    if len(parts) >= 6:
                        mode = parts[0] if parts[0] in ("brush", "erase") else "brush"
                        stroke_type = parts[1] if parts[1] in ("free", "box", "square") else "free"
                        stroke_size = int(float(parts[2]))
                        stroke_opacity = float(parts[3])
                        stroke_color = parts[4]
                        if len(parts) >= 7 and parts[5] in ("1", "2", "3", "4", "5", "6") and "," not in parts[5]:
                            marker_id = parts[5]
                            points_str = ":".join(parts[6:])
                        else:
                            points_str = ":".join(parts[5:])
                    elif len(parts) >= 2:
                        if parts[0] in ("brush", "erase"):
                            mode = parts[0]
                            points_str = ":".join(parts[1:])
                        elif parts[0] in ("free", "box", "square"):
                            stroke_type = parts[0]
                            points_str = ":".join(parts[1:])

                radius = max(1, stroke_size // 2)
                path_points = self._parse_path_points(points_str)
                if not path_points:
                    continue

                color_type = color_mapping.get(stroke_color, "default")
                x_coords = [point[0] for point in path_points]
                y_coords = [point[1] for point in path_points]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                if marker_id is not None and stroke_type == "square" and mode != "erase":
                    valid_min_x = max(0, min_x)
                    valid_max_x = min(width - 1, max_x)
                    valid_min_y = max(0, min_y)
                    valid_max_y = min(height - 1, max_y)
                    if valid_max_x > valid_min_x and valid_max_y > valid_min_y:
                        marker_annotations.append(
                            {
                                "id": marker_id,
                                "min_x": int(valid_min_x),
                                "max_x": int(valid_max_x),
                                "min_y": int(valid_min_y),
                                "max_y": int(valid_max_y),
                            }
                        )
                    continue

                if stroke_type == "square":
                    self._apply_square(
                        mode,
                        color_type,
                        stroke_opacity,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        width,
                        height,
                        black_mask_np,
                        white_mask_np,
                        red_mask_np,
                        green_mask_np,
                        blue_mask_np,
                        gray_mask_np,
                    )
                elif stroke_type == "box":
                    self._apply_box(
                        mode,
                        color_type,
                        stroke_opacity,
                        stroke_size,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                        width,
                        height,
                        black_mask_np,
                        white_mask_np,
                        red_mask_np,
                        green_mask_np,
                        blue_mask_np,
                        gray_mask_np,
                    )
                else:
                    self._apply_free_path(
                        mode,
                        color_type,
                        path_points,
                        radius,
                        black_mask_np,
                        white_mask_np,
                        red_mask_np,
                        green_mask_np,
                        blue_mask_np,
                        gray_mask_np,
                    )

            black_mask[0] = torch.from_numpy(black_mask_np)
            white_mask[0] = torch.from_numpy(white_mask_np)
            red_mask[0] = torch.from_numpy(red_mask_np)
            green_mask[0] = torch.from_numpy(green_mask_np)
            blue_mask[0] = torch.from_numpy(blue_mask_np)
            gray_mask[0] = torch.from_numpy(gray_mask_np)
        except Exception as error:
            print(f"[Shaobkj_QuickMark] Failed to parse brush_data: {error}")

    def _parse_path_points(self, points_str):
        path_points = []
        for point_str in points_str.split(";"):
            if not point_str.strip():
                continue
            try:
                coords = point_str.split(",", 1)
                if len(coords) == 2:
                    x = int(float(coords[0]))
                    y = int(float(coords[1]))
                    path_points.append((x, y))
            except (ValueError, IndexError):
                continue
        return path_points

    def _apply_square(
        self,
        mode,
        color_type,
        stroke_opacity,
        min_x,
        max_x,
        min_y,
        max_y,
        width,
        height,
        black_mask_np,
        white_mask_np,
        red_mask_np,
        green_mask_np,
        blue_mask_np,
        gray_mask_np,
    ):
        valid_min_x = max(0, min_x)
        valid_max_x = min(width, max_x + 1)
        valid_min_y = max(0, min_y)
        valid_max_y = min(height, max_y + 1)

        if valid_max_x <= valid_min_x or valid_max_y <= valid_min_y:
            return

        all_masks = (black_mask_np, white_mask_np, red_mask_np, green_mask_np, blue_mask_np, gray_mask_np)
        target_mask = self._get_target_mask(color_type, *all_masks)

        if mode == "erase":
            for mask in all_masks:
                mask[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
            return

        if target_mask is not None:
            target_mask[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity

    def _apply_box(
        self,
        mode,
        color_type,
        stroke_opacity,
        stroke_size,
        min_x,
        max_x,
        min_y,
        max_y,
        width,
        height,
        black_mask_np,
        white_mask_np,
        red_mask_np,
        green_mask_np,
        blue_mask_np,
        gray_mask_np,
    ):
        x0 = max(0, min_x)
        x1 = min(width, max_x + 1)
        y0 = max(0, min_y)
        y1 = min(height, max_y + 1)

        if x1 <= x0 or y1 <= y0:
            return

        thickness = max(1, int(stroke_size))
        top_y1 = min(y1, y0 + thickness)
        bottom_y0 = max(y0, y1 - thickness)
        left_x1 = min(x1, x0 + thickness)
        right_x0 = max(x0, x1 - thickness)

        edges = [
            (slice(y0, top_y1), slice(x0, x1)),
            (slice(bottom_y0, y1), slice(x0, x1)),
            (slice(y0, y1), slice(x0, left_x1)),
            (slice(y0, y1), slice(right_x0, x1)),
        ]

        all_masks = (black_mask_np, white_mask_np, red_mask_np, green_mask_np, blue_mask_np, gray_mask_np)
        target_mask = self._get_target_mask(color_type, *all_masks)

        if mode == "erase":
            for mask in all_masks:
                for ys, xs in edges:
                    mask[ys, xs] = 0.0
            return

        if target_mask is not None:
            for ys, xs in edges:
                target_mask[ys, xs] = stroke_opacity

    def _apply_free_path(
        self,
        mode,
        color_type,
        path_points,
        radius,
        black_mask_np,
        white_mask_np,
        red_mask_np,
        green_mask_np,
        blue_mask_np,
        gray_mask_np,
    ):
        all_masks = (black_mask_np, white_mask_np, red_mask_np, green_mask_np, blue_mask_np, gray_mask_np)
        target_mask = self._get_target_mask(color_type, *all_masks)

        for index, (x, y) in enumerate(path_points):
            if index > 0:
                prev_x, prev_y = path_points[index - 1]
                if mode == "erase":
                    for mask in all_masks:
                        self._erase_line(mask, prev_x, prev_y, x, y, radius)
                elif target_mask is not None:
                    self._draw_line(target_mask, prev_x, prev_y, x, y, radius)
            else:
                if mode == "erase":
                    for mask in all_masks:
                        self._erase_circle(mask, x, y, radius)
                elif target_mask is not None:
                    self._draw_circle(target_mask, x, y, radius)

    def _get_target_mask(self, color_type, black_mask_np, white_mask_np, red_mask_np, green_mask_np, blue_mask_np, gray_mask_np):
        if color_type == "black":
            return black_mask_np
        if color_type == "white":
            return white_mask_np
        if color_type == "red":
            return red_mask_np
        if color_type == "green":
            return green_mask_np
        if color_type == "blue":
            return blue_mask_np
        if color_type == "gray":
            return gray_mask_np
        return None

    def _draw_marker_annotations(self, sum_image, marker_annotations):
        font_cache = {}
        pil_font_path = os.path.join(os.path.dirname(ImageFont.__file__), "fonts", "DejaVuSans.ttf")

        def get_font(font_size):
            font_size = int(font_size)
            cached_font = font_cache.get(font_size)
            if cached_font is not None:
                return cached_font
            font = None
            for candidate in ("arial.ttf", "DejaVuSans.ttf", pil_font_path):
                try:
                    font = ImageFont.truetype(candidate, size=font_size)
                    break
                except Exception:
                    font = None
            if font is None:
                font = ImageFont.load_default()
            font_cache[font_size] = font
            return font

        for batch_index in range(sum_image.shape[0]):
            image_np = (sum_image[batch_index].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np, mode="RGB")
            draw = ImageDraw.Draw(image_pil)

            for marker in marker_annotations:
                x0 = int(marker["min_x"])
                y0 = int(marker["min_y"])
                x1 = int(marker["max_x"])
                y1 = int(marker["max_y"])
                side = max(1, min(abs(x1 - x0), abs(y1 - y0)))
                font = get_font(max(12, int(side * 0.6)))
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
                text = str(marker["id"])
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except Exception:
                    text_width, text_height = font.getsize(text)
                text_x = x0 + max(0, (x1 - x0 - text_width) // 2)
                text_y = y0 + max(0, (y1 - y0 - text_height) // 2)
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

            image_out = np.array(image_pil).astype(np.float32) / 255.0
            sum_image[batch_index] = torch.from_numpy(image_out).to(sum_image.device)

        return sum_image

    def _draw_circle(self, mask, x, y, radius):
        height, width = mask.shape
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)

        if x_max <= x_min or y_max <= y_min:
            return

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        radius_sq = radius * radius

        mask[y_min:y_max, x_min:x_max] = np.maximum(
            mask[y_min:y_max, x_min:x_max],
            (dist_sq <= radius_sq).astype(np.float32),
        )

    def _draw_line(self, mask, x1, y1, x2, y2, radius):
        if x1 == x2 and y1 == y2:
            self._draw_circle(mask, x1, y1, radius)
            return

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        step_size = max(1, radius // 3) if radius > 10 else 1
        steps = max(1, int(length / step_size) + 1)

        if steps <= 0:
            self._draw_circle(mask, x1, y1, radius)
            return

        t_values = np.linspace(0, 1, steps + 1)
        x_coords = (x1 + dx * t_values).astype(np.int32)
        y_coords = (y1 + dy * t_values).astype(np.int32)

        height, width = mask.shape
        valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if len(x_coords) > 0:
            coords = np.column_stack((y_coords, x_coords))
            unique_coords = np.unique(coords, axis=0)
            for y, x in unique_coords:
                self._draw_circle(mask, int(x), int(y), radius)

    def _erase_circle(self, mask, x, y, radius):
        height, width = mask.shape
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)

        if x_max <= x_min or y_max <= y_min:
            return

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
        radius_sq = radius * radius
        erase_mask = dist_sq <= radius_sq

        mask[y_min:y_max, x_min:x_max] = np.where(
            erase_mask,
            0.0,
            mask[y_min:y_max, x_min:x_max],
        )

    def _erase_line(self, mask, x1, y1, x2, y2, radius):
        if x1 == x2 and y1 == y2:
            self._erase_circle(mask, x1, y1, radius)
            return

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        step_size = max(1, radius // 3) if radius > 10 else 1
        steps = max(1, int(length / step_size) + 1)

        if steps <= 0:
            self._erase_circle(mask, x1, y1, radius)
            return

        t_values = np.linspace(0, 1, steps + 1)
        x_coords = (x1 + dx * t_values).astype(np.int32)
        y_coords = (y1 + dy * t_values).astype(np.int32)

        height, width = mask.shape
        valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        if len(x_coords) > 0:
            coords = np.column_stack((y_coords, x_coords))
            unique_coords = np.unique(coords, axis=0)
            for y, x in unique_coords:
                self._erase_circle(mask, int(x), int(y), radius)
