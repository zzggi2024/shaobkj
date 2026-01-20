from .node_api_generator import Shaobkj_APINode, Shaobkj_APINode_Batch
from .node_reverse import Shaobkj_Reverse_Node
from .node_video import Shaobkj_Sora_Video
from .node_veo_video import Shaobkj_Veo_Video
from .node_concurrent_image_edit import Shaobkj_ConcurrentImageEdit, Shaobkj_ConcurrentImageEdit_Sender, Shaobkj_ConcurrentImageEdit_Receiver, Shaobkj_Load_Batch_Images
from .node_http_transfer import Shaobkj_HTTP_Load_Image, Shaobkj_HTTP_Send_Image

NODE_CLASS_MAPPINGS = {
    "Shaobkj_APINode": Shaobkj_APINode,
    "Shaobkj_APINode_Batch": Shaobkj_APINode_Batch,
    "Shaobkj_Reverse_Node": Shaobkj_Reverse_Node,
    "Shaobkj_Sora_Video": Shaobkj_Sora_Video,
    "Shaobkj_Veo_Video": Shaobkj_Veo_Video,
    "Shaobkj_ConcurrentImageEdit": Shaobkj_ConcurrentImageEdit,
    "Shaobkj_ConcurrentImageEdit_Sender": Shaobkj_ConcurrentImageEdit_Sender,
    "Shaobkj_ConcurrentImageEdit_Receiver": Shaobkj_ConcurrentImageEdit_Receiver,
    "Shaobkj_Load_Batch_Images": Shaobkj_Load_Batch_Images,
    "Shaobkj_HTTP_Load_Image": Shaobkj_HTTP_Load_Image,
    "Shaobkj_HTTP_Send_Image": Shaobkj_HTTP_Send_Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_APINode": "🤖图像生成",
    "Shaobkj_APINode_Batch": "🤖并发-文本-图像生成",
    "Shaobkj_Reverse_Node": "🤖 Shaobkj 反推",
    "Shaobkj_Sora_Video": "🤖 Shaobkj -Sora视频",
    "Shaobkj_Veo_Video": "🤖 Shaobkj -Veo视频",
    "Shaobkj_ConcurrentImageEdit": "🤖并发-图像编辑 (Legacy)",
    "Shaobkj_ConcurrentImageEdit_Sender": "🤖并发-编辑-发送端",
    "Shaobkj_ConcurrentImageEdit_Receiver": "🤖并发-编辑-接收端",
    "Shaobkj_Load_Batch_Images": "🤖批量加载图像(路径)",
    "Shaobkj_HTTP_Load_Image": "🤖本地桥接-加载图片",
    "Shaobkj_HTTP_Send_Image": "🤖本地桥接-发送图片",
}

print("[ComfyUI-shaobkj] Node code loaded. Please restart ComfyUI if you see this message for the first time.")
