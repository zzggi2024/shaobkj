class Shaobkj_Text_Process:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "process_text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"default": "", "multiline": True, "tooltip": "输入待处理文本"}),
                "模式选择": (["去空行", "去字符", "去空行+去字符"], {"default": "去空行", "tooltip": "文本处理模式；可选同时生效"}),
                "去除内容": ("STRING", {"default": "", "multiline": True, "tooltip": "去字符模式下生效；输入需要去除的符号或文本"}),
            }
        }

    def process_text(self, 文本, 模式选择, 去除内容):
        source = str(文本) if 文本 is not None else ""
        mode = 模式选择 if 模式选择 is not None else "去空行"
        remove_content = str(去除内容) if 去除内容 is not None else ""
        result = source

        selected_modes = set()
        if isinstance(mode, (list, tuple, set)):
            for item in mode:
                if item is not None:
                    selected_modes.add(str(item))
        else:
            selected_modes.add(str(mode))

        if "去空行+去字符" in selected_modes:
            selected_modes.add("去空行")
            selected_modes.add("去字符")

        if "去空行" in selected_modes:
            lines = result.splitlines()
            kept = [line for line in lines if line.strip() != ""]
            result = "\n".join(kept)

        if "去字符" in selected_modes and remove_content != "":
            result = result.replace(remove_content, "")

        return (result,)
