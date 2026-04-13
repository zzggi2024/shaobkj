import hashlib
import os

import torch
from aiohttp import web
from server import PromptServer

class AnyType(str):
    """Wildcard type that matches everything."""
    def __ne__(self, __value: object) -> bool:
        return False
    def __eq__(self, __value: object) -> bool:
        return True

any_type = AnyType("*")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif"}
LOOP_TRIGGER_STATE = {}


def _normalize_folder_path(folder_path):
    value = str(folder_path or "").strip().strip('"').strip("'")
    if not value:
        return ""
    return os.path.normpath(os.path.expandvars(os.path.expanduser(value)))


def _scan_image_folder(folder_path):
    normalized_path = _normalize_folder_path(folder_path)
    if not normalized_path:
        return {
            "path": "",
            "exists": False,
            "is_dir": False,
            "total": 0,
            "signature": "",
        }

    if not os.path.exists(normalized_path):
        return {
            "path": normalized_path,
            "exists": False,
            "is_dir": False,
            "total": 0,
            "signature": "",
        }

    if not os.path.isdir(normalized_path):
        return {
            "path": normalized_path,
            "exists": True,
            "is_dir": False,
            "total": 0,
            "signature": "",
        }

    digest = hashlib.sha1()
    total = 0
    with os.scandir(normalized_path) as entries:
        image_entries = []
        for entry in entries:
            if not entry.is_file():
                continue
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            stat = entry.stat()
            image_entries.append((entry.name, int(stat.st_size), int(stat.st_mtime_ns)))

    image_entries.sort(key=lambda item: item[0].lower())
    for name, size, mtime_ns in image_entries:
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(str(size).encode("utf-8"))
        digest.update(str(mtime_ns).encode("utf-8"))
        total += 1

    return {
        "path": normalized_path,
        "exists": True,
        "is_dir": True,
        "total": total,
        "signature": digest.hexdigest(),
    }


def _send_loop_feedback(unique_id, widget_name, value):
    if unique_id is None:
        return
    PromptServer.instance.send_sync(
        "shaobkj.loop_trigger.feedback",
        {"node_id": str(unique_id), "widget_name": widget_name, "value": value},
    )


def _send_loop_add_queue():
    PromptServer.instance.send_sync("shaobkj.loop_trigger.add_queue", {})


@PromptServer.instance.routes.post("/shaobkj/loop_trigger/scan")
async def shaobkj_loop_trigger_scan(request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    result = _scan_image_folder(payload.get("path", ""))
    return web.json_response({"status": "success", **result})


@PromptServer.instance.routes.post("/shaobkj/loop_trigger/initialize")
async def shaobkj_loop_trigger_initialize(request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    unique_id = str(payload.get("unique_id", "") or "")
    result = _scan_image_folder(payload.get("path", ""))
    initialized = bool(result.get("exists")) and bool(result.get("is_dir"))
    if unique_id:
        LOOP_TRIGGER_STATE[unique_id] = {"initialized": initialized}
    return web.json_response({"status": "success", "initialized": initialized, **result})

class Shaobkj_ForLoop_Start:
    """
    Optimized For Loop Start node using Batch Lists.
    Generates a list of indices to drive downstream nodes efficiently without graph expansion.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "循环次数": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "循环总次数 (Batch Size)"}),
                "起始索引": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "起始索引值"}),
            },
            "optional": {
                "初始值": (any_type, {"tooltip": "可选：传入初始值（仅透传）"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", any_type)
    RETURN_NAMES = ("索引", "总数", "初始值")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "execute"
    CATEGORY = "🤖shaobkj-APIbox/Logic"

    def execute(self, 循环次数, 起始索引, 初始值=None):
        # Generate indices list [start, ..., start+total-1]
        indices = [起始索引 + i for i in range(循环次数)]
        print(f"[ComfyUI-shaobkj] Loop Start: Generated batch of {len(indices)} indices starting from {起始索引}")
        return (indices, 循环次数, 初始值)

class Shaobkj_ForLoop_End:
    """
    Optimized For Loop End node using Batch Lists.
    Collects results from a batched execution into a single list.
    Acts as a synchronization point for the loop.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入数据": (any_type, {"tooltip": "连接循环体内的输出结果"}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("输出列表",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "🤖shaobkj-APIbox/Logic"

    def execute(self, 输入数据):
        # 输入数据 is already a list of results from the batch execution
        count = len(输入数据) if isinstance(输入数据, list) else 1
        print(f"[ComfyUI-shaobkj] Loop End: Collected {count} items")
        return (输入数据,)


class Shaobkj_Loop_Trigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文件夹路径": ("STRING", {"default": "", "multiline": False, "tooltip": "图片文件夹路径；首次运行会自动初始化，也可手动初始化"}),
                "强制循环数": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "默认跟随总数；手动填写后按该值循环"}),
                "总数": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "初始化或自动检测后同步图片总数"}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "触发", "label_off": "不触发", "tooltip": "是否继续自动触发队列"}),
                "当前执行编号": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "实时显示当前执行编号"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("执行计数", "总数")
    FUNCTION = "execute"
    CATEGORY = "🤖shaobkj-APIbox/Logic"
    OUTPUT_NODE = True

    def execute(self, 文件夹路径, 强制循环数, 总数, mode, 当前执行编号, unique_id):
        scan_result = _scan_image_folder(文件夹路径)
        actual_total = int(scan_result["total"])
        stored_total = int(总数) if 总数 is not None else 0

        if actual_total != stored_total:
            _send_loop_feedback(unique_id, "总数", actual_total)
            _send_loop_feedback(unique_id, "强制循环数", actual_total)

        state_key = str(unique_id)
        state = LOOP_TRIGGER_STATE.get(state_key, {"initialized": False})
        initialized = bool(state.get("initialized", False))
        if actual_total <= 0:
            LOOP_TRIGGER_STATE[state_key] = {"initialized": False}
            _send_loop_feedback(unique_id, "当前执行编号", 0)
            return (0, actual_total)

        if not initialized:
            LOOP_TRIGGER_STATE[state_key] = {"initialized": True}
            _send_loop_feedback(unique_id, "当前执行编号", 0)

        loop_total = int(强制循环数) if 强制循环数 is not None else 0
        if loop_total <= 0:
            loop_total = actual_total
        loop_total = max(1, loop_total)

        current_value = int(当前执行编号) if 当前执行编号 is not None else 0
        current_value = min(max(current_value, 0), loop_total - 1)
        _send_loop_feedback(unique_id, "当前执行编号", current_value)

        if bool(mode):
            if current_value < loop_total - 1:
                _send_loop_feedback(unique_id, "当前执行编号", current_value + 1)
                _send_loop_add_queue()
            else:
                _send_loop_feedback(unique_id, "当前执行编号", 0)

        return (current_value, actual_total)
