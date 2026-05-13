class Shaobkj_ZeroOneFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "步进": (["0.1", "0.01"], {"default": "0.1"}),
                "数值": ("FLOAT", {"default": 0.1, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("浮点数",)
    FUNCTION = "get_value"
    CATEGORY = "🤖shaobkj-APIbox/实用工具"

    def get_value(self, 步进, 数值):
        step_value = 0.01 if str(步进) == "0.01" else 0.1
        min_value = step_value
        clamped_value = max(min_value, min(1.0, float(数值)))
        digits = 2 if step_value == 0.01 else 1
        rounded_value = round(clamped_value / step_value) * step_value
        rounded_value = max(min_value, min(1.0, rounded_value))
        return (round(rounded_value, digits),)
