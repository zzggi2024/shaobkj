from .node_api_generator import Shaobkj_APINode, Shaobkj_APINode_Batch
from .node_reverse import Shaobkj_Reverse_Node
from .node_video import Shaobkj_Sora_Video
from .node_veo_video import Shaobkj_Veo_Video
from .node_concurrent_image_edit import Shaobkj_ConcurrentImageEdit_Sender, Shaobkj_Load_Batch_Images
from .node_jimeng_avatar import Shaobkj_Jimeng_Avatar

NODE_CLASS_MAPPINGS = {
    "Shaobkj_APINode": Shaobkj_APINode,
    "Shaobkj_APINode_Batch": Shaobkj_APINode_Batch,
    "Shaobkj_Reverse_Node": Shaobkj_Reverse_Node,
    "Shaobkj_Sora_Video": Shaobkj_Sora_Video,
    "Shaobkj_Veo_Video": Shaobkj_Veo_Video,
    "Shaobkj_ConcurrentImageEdit_Sender": Shaobkj_ConcurrentImageEdit_Sender,
    "Shaobkj_Load_Batch_Images": Shaobkj_Load_Batch_Images,
    "Shaobkj_Jimeng_Avatar": Shaobkj_Jimeng_Avatar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_APINode": "🤖图像生成",
    "Shaobkj_APINode_Batch": "🤖并发-文本-图像生成",
    "Shaobkj_Reverse_Node": "🤖 Shaobkj 反推",
    "Shaobkj_Sora_Video": "🤖 Shaobkj -Sora视频",
    "Shaobkj_Veo_Video": "🤖 Shaobkj -Veo视频",
    "Shaobkj_ConcurrentImageEdit_Sender": "🤖并发-编辑-发送端",
    "Shaobkj_Load_Batch_Images": "🤖批量加载图像(路径)",
    "Shaobkj_Jimeng_Avatar": "🤖即梦数字人（待测试）",
}

print("[ComfyUI-shaobkj] Node code loaded. Please restart ComfyUI if you see this message for the first time.")
