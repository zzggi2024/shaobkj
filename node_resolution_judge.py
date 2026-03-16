import torch
import torch.nn.functional as F


class Shaobkj_ResolutionJudge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "输入待判断图像；推荐：连接图像输出"}),
                "阈值": ("INT", {"default": 1024, "min": 1, "max": 32768, "step": 1, "tooltip": "手动输入长边阈值像素；推荐：1024"}),
                "阈值缩放边": (["长边", "短边"], {"default": "长边", "tooltip": "按长边或短边等比缩放到阈值"}),
                "保持原图": (["关闭", "打开"], {"default": "关闭", "tooltip": "打开时，未超出阈值仅做取整不缩放"}),
                "取整倍数": (["8", "16", "32", "64"], {"default": "8", "tooltip": "输出宽高需被该值整除；推荐：8"}),
                "取整模式": (["向上", "向下"], {"default": "向上", "tooltip": "尺寸对齐到倍数时的取整方向；推荐：向上"}),
            },
            "optional": {
                "遮罩": ("MASK", {"tooltip": "可选遮罩；若提供将按同尺寸同步缩放"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "IMAGE", "MASK")
    RETURN_NAMES = ("是否超出", "阈值图像", "阈值遮罩")
    FUNCTION = "judge_resolution"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def judge_resolution(self, 图像, 阈值, 阈值缩放边, 保持原图, 取整倍数, 取整模式, 遮罩=None):
        if not isinstance(图像, torch.Tensor):
            raise ValueError("❌ 错误：输入图像类型无效")

        t = 图像
        if t.dim() == 3:
            t = t.unsqueeze(0)

        if t.dim() != 4:
            raise ValueError("❌ 错误：输入图像张量维度必须为 [B,H,W,C] 或 [H,W,C]")

        h = int(t.shape[1])
        w = int(t.shape[2])
        threshold_px = int(阈值)
        if threshold_px <= 0:
            raise ValueError("❌ 错误：阈值必须大于 0")
        multiple = int(str(取整倍数))
        if multiple <= 0:
            raise ValueError("❌ 错误：取整倍数必须大于 0")
        use_short_edge = str(阈值缩放边) == "短边"
        reference_edge = min(w, h) if use_short_edge else max(w, h)
        exceeded = bool(reference_edge > threshold_px)
        keep_original = str(保持原图) == "打开"

        if keep_original and not exceeded:
            new_w = w
            new_h = h
        else:
            if use_short_edge:
                if w <= h:
                    new_w = threshold_px
                    new_h = max(1, int(round(h * threshold_px / w)))
                else:
                    new_h = threshold_px
                    new_w = max(1, int(round(w * threshold_px / h)))
            else:
                if w >= h:
                    new_w = threshold_px
                    new_h = max(1, int(round(h * threshold_px / w)))
                else:
                    new_h = threshold_px
                    new_w = max(1, int(round(w * threshold_px / h)))

        def align_size(value, m, mode):
            if mode == "向下":
                aligned = (value // m) * m
                return max(m, aligned)
            aligned = ((value + m - 1) // m) * m
            return max(m, aligned)

        new_w = align_size(int(new_w), multiple, str(取整模式))
        new_h = align_size(int(new_h), multiple, str(取整模式))

        t_bchw = t.permute(0, 3, 1, 2)
        resized_bchw = F.interpolate(t_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        resized_bhwc = resized_bchw.permute(0, 2, 3, 1)

        b = int(t.shape[0])
        if 遮罩 is None:
            if b == 1:
                resized_mask = torch.zeros((new_h, new_w), dtype=torch.float32, device=t.device)
            else:
                resized_mask = torch.zeros((b, new_h, new_w), dtype=torch.float32, device=t.device)
        else:
            if not isinstance(遮罩, torch.Tensor):
                raise ValueError("❌ 错误：遮罩类型无效")
            m = 遮罩
            if m.dim() == 2:
                m = m.unsqueeze(0)
            elif m.dim() == 3:
                pass
            elif m.dim() == 4 and int(m.shape[-1]) == 1:
                m = m.squeeze(-1)
            else:
                raise ValueError("❌ 错误：遮罩张量维度必须为 [H,W] 或 [B,H,W]")

            mb = int(m.shape[0])
            if mb == 1 and b > 1:
                m = m.expand(b, -1, -1)
            elif mb != b:
                raise ValueError("❌ 错误：遮罩批次数与图像不一致")

            m_bchw = m.unsqueeze(1)
            resized_mask_batch = F.interpolate(m_bchw, size=(new_h, new_w), mode="nearest").squeeze(1)
            resized_mask = resized_mask_batch[0] if b == 1 else resized_mask_batch

        return (exceeded, resized_bhwc, resized_mask)
