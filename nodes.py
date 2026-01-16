from .node_api_generator import Shaobkj_APINode
from .node_reverse import Shaobkj_Reverse_Node
from .node_video import Shaobkj_Sora_Video
from .node_veo_video import Shaobkj_Veo_Video

NODE_CLASS_MAPPINGS = {
    "Shaobkj_APINode": Shaobkj_APINode,
    "Shaobkj_Reverse_Node": Shaobkj_Reverse_Node,
    "Shaobkj_Sora_Video": Shaobkj_Sora_Video,
    "Shaobkj_Veo_Video": Shaobkj_Veo_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_APINode": "🤖 Shaobkj 图像生成",
    "Shaobkj_Reverse_Node": "🤖 Shaobkj 反推",
    "Shaobkj_Sora_Video": "🤖 Shaobkj -Sora视频",
    "Shaobkj_Veo_Video": "🤖 Shaobkj -Veo视频",
}

print("[ComfyUI-shaobkj] Node code loaded. Please restart ComfyUI if you see this message for the first time.")
