import torch

class AnyType(str):
    """Wildcard type that matches everything."""
    def __ne__(self, __value: object) -> bool:
        return False
    def __eq__(self, __value: object) -> bool:
        return True

any_type = AnyType("*")

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
