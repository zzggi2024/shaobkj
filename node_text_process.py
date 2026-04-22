import re

from comfy_execution.graph_utils import ExecutionBlocker
from server import PromptServer


NUMBER_PREFIX_PATTERN = re.compile(r"^\s*(?:\d+|[一二三四五六七八九十百千万零〇两]+)(?:\s*[\.．、,，:：;；\)\]）】>\-—_]+|\s+)+")
REMOVE_PREFIX_NUMBER_ALIASES = {"去除开头编号", "删除开头编号", "去掉开头编号"}


def _remove_line_prefix_numbers(text):
    lines = text.splitlines()
    cleaned_lines = [NUMBER_PREFIX_PATTERN.sub("", line, count=1) for line in lines]
    return "\n".join(cleaned_lines)


def _parse_remove_items(text):
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item and item.strip()]


def _send_text_process_feedback(unique_id, widget_name, value):
    if unique_id is None:
        return
    PromptServer.instance.send_sync(
        "shaobkj.node_feedback",
        {"node_id": str(unique_id), "widget_name": widget_name, "value": value},
    )


class Shaobkj_Text_Process:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "process_text"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("文本", "输出列表数", "当前执行编号")
    OUTPUT_IS_LIST = (True, False, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"default": "", "multiline": True, "tooltip": "输入待处理文本"}),
                "模式选择": (["去空行", "去字符", "去空行+去字符"], {"default": "去空行", "tooltip": "文本处理模式；可选同时生效"}),
                "去除内容": ("STRING", {"default": "", "multiline": True, "placeholder": "去除内容，多个用英文逗号,隔开", "tooltip": "去字符模式下生效；多个用英文逗号,隔开；支持：去除开头编号"}),
                "列表": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后输出列表形式，并输出列表数"}),
                "计数开始": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "计数结束": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "触发", "label_off": "不触发", "tooltip": "是否按编号自动继续触发队列"}),
                "当前执行编号状态": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def process_text(self, 文本, 模式选择, 去除内容, 列表, 计数开始, 计数结束, mode, 当前执行编号状态, unique_id):
        source = str(文本) if 文本 is not None else ""
        selected_mode = 模式选择 if 模式选择 is not None else "去空行"
        remove_content = str(去除内容) if 去除内容 is not None else ""
        list_mode = bool(列表)
        start_value = max(0, int(计数开始) if 计数开始 is not None else 0)
        end_value = int(计数结束) if 计数结束 is not None else 0
        current_state = int(当前执行编号状态) if 当前执行编号状态 is not None else start_value
        trigger_mode = bool(mode)
        result = source

        selected_modes = set()
        if isinstance(selected_mode, (list, tuple, set)):
            for item in selected_mode:
                if item is not None:
                    selected_modes.add(str(item))
        else:
            selected_modes.add(str(selected_mode))

        if "去空行+去字符" in selected_modes:
            selected_modes.add("去空行")
            selected_modes.add("去字符")

        if "去空行" in selected_modes:
            lines = result.splitlines()
            kept = [line for line in lines if line.strip() != ""]
            result = "\n".join(kept)

        if "去字符" in selected_modes and remove_content != "":
            remove_items = _parse_remove_items(remove_content)
            if any(item in REMOVE_PREFIX_NUMBER_ALIASES for item in remove_items):
                result = _remove_line_prefix_numbers(result)
            for item in remove_items:
                if item not in REMOVE_PREFIX_NUMBER_ALIASES:
                    result = result.replace(item, "")

        result_list = result.split("\n") if result != "" else []
        total_count = len(result_list) if list_mode else 0
        output_count = 0
        current_index = 0
        output_list = result_list

        if list_mode and total_count > 0:
            start_value = min(start_value, total_count - 1)
            if end_value <= 0:
                end_value = total_count
            end_value = min(max(end_value, start_value + 1), total_count)
            output_count = max(0, end_value - start_value)
            max_index = end_value - 1
            current_index = min(max(start_value, current_state), max_index)
            _send_text_process_feedback(unique_id, "当前执行编号状态", current_index)
            if trigger_mode:
                output_list = [result_list[current_index]]
                if current_index < max_index:
                    _send_text_process_feedback(unique_id, "当前执行编号状态", current_index + 1)
                    PromptServer.instance.send_sync("shaobkj.add_queue", {})
                else:
                    _send_text_process_feedback(unique_id, "当前执行编号状态", start_value)
            else:
                _send_text_process_feedback(unique_id, "当前执行编号状态", current_index)
        else:
            _send_text_process_feedback(unique_id, "当前执行编号状态", 0)

        if list_mode:
            return (output_list, output_count, current_index)

        return ([result], ExecutionBlocker(None), ExecutionBlocker(None))


class Shaobkj_InfinitePromptJoin:
    CATEGORY = "🤖shaobkj-APIbox/实用工具"
    FUNCTION = "join_prompts"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输出模式": (["普通拼接", "规范化拼接"], {"default": "普通拼接", "tooltip": "普通拼接按分隔符直接连接；规范化拼接会去空行并按行尾符号逐行输出"}),
                "分隔符": ("STRING", {"default": ", ", "multiline": False, "tooltip": "用于连接多个提示词"}),
                "行尾符号": ("STRING", {"default": "；", "multiline": False, "tooltip": "规范化拼接时，自动补到每行结尾"}),
            },
            "optional": {
                "提示词1": ("STRING", {"forceInput": True, "default": "", "multiline": True}),
                "提示词2": ("STRING", {"forceInput": True, "default": "", "multiline": True}),
            },
        }

    @staticmethod
    def _collect_prompt_items(kwargs):
        prompt_items = []
        prompt_keys = []

        for key in kwargs.keys():
            if isinstance(key, str) and key.startswith("提示词"):
                suffix = key.replace("提示词", "", 1)
                if suffix.isdigit():
                    prompt_keys.append((int(suffix), key))

        prompt_keys.sort(key=lambda item: item[0])

        for _, key in prompt_keys:
            value = kwargs.get(key)
            if value is None:
                continue
            text = str(value)
            for line in text.splitlines():
                cleaned = line.strip()
                if cleaned != "":
                    prompt_items.append(cleaned)

        return prompt_items

    @staticmethod
    def _normalize_prompt_items(prompt_items, line_suffix):
        normalized_items = []
        suffix = str(line_suffix or "").strip()

        for item in prompt_items:
            text = str(item).strip()
            if text == "":
                continue
            if suffix and not text.endswith(suffix):
                text = f"{text}{suffix}"
            normalized_items.append(text)

        return normalized_items

    def join_prompts(self, 输出模式="普通拼接", 分隔符=", ", 行尾符号="；", **kwargs):
        prompt_items = self._collect_prompt_items(kwargs)

        if 输出模式 == "规范化拼接":
            normalized_items = self._normalize_prompt_items(prompt_items, 行尾符号)
            return ("\n".join(normalized_items),)

        return (str(分隔符).join(prompt_items),)
