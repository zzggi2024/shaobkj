from .node_api_generator import Shaobkj_APINode, Shaobkj_APINode_Batch
from .node_video import Shaobkj_Sora_Video
from .node_veo_video import Shaobkj_Veo_Video
from .node_concurrent_image_edit import Shaobkj_ConcurrentImageEdit_Sender, Shaobkj_GroupedConcurrentImageEdit, Shaobkj_Load_Image_Path, Shaobkj_Load_Batch_Images, Shaobkj_Image_Save, Shaobkj_Fixed_Seed
from .node_llm_app import Shaobkj_LLM_App, Shaobkj_NanoBanana_Prompt
from .node_loop import Shaobkj_ForLoop_Start, Shaobkj_ForLoop_End
from .node_load_image_list import Shaobkj_LoadImageListFromDir
from .node_resolution_judge import Shaobkj_ResolutionJudge, Shaobkj_GetEdgeLength
from .node_florence2_fast_prompt import Shaobkj_Load_Florence2_Model, Shaobkj_Florence2_Fast_Prompt
from .node_seamless_pattern import Shaobkj_SeamlessPattern
from .node_free_color import Shaobkj_FreeColor
from .node_text_process import Shaobkj_Text_Process

NODE_CLASS_MAPPINGS = {
    "Shaobkj_APINode": Shaobkj_APINode,
    "Shaobkj_APINode_Batch": Shaobkj_APINode_Batch,
    "Shaobkj_Sora_Video": Shaobkj_Sora_Video,
    "Shaobkj_Veo_Video": Shaobkj_Veo_Video,
    "Shaobkj_ConcurrentImageEdit_Sender": Shaobkj_ConcurrentImageEdit_Sender,
    "Shaobkj_GroupedConcurrentImageEdit": Shaobkj_GroupedConcurrentImageEdit,
    "Shaobkj_Load_Image_Path": Shaobkj_Load_Image_Path,
    "Shaobkj_Load_Batch_Images": Shaobkj_Load_Batch_Images,
    "Shaobkj_Image_Save": Shaobkj_Image_Save,
    "Shaobkj_Fixed_Seed": Shaobkj_Fixed_Seed,
    "Shaobkj_LLM_App": Shaobkj_LLM_App,
    "Shaobkj_ForLoop_Start": Shaobkj_ForLoop_Start,
    "Shaobkj_ForLoop_End": Shaobkj_ForLoop_End,
    "Shaobkj_NanoBanana_Prompt": Shaobkj_NanoBanana_Prompt,
    "Shaobkj_LoadImageListFromDir": Shaobkj_LoadImageListFromDir,
    "Shaobkj_ResolutionJudge": Shaobkj_ResolutionJudge,
    "Shaobkj_GetEdgeLength": Shaobkj_GetEdgeLength,
    "Shaobkj_Load_Florence2_Model": Shaobkj_Load_Florence2_Model,
    "Shaobkj_Florence2_Fast_Prompt": Shaobkj_Florence2_Fast_Prompt,
    "Shaobkj_SeamlessPattern": Shaobkj_SeamlessPattern,
    "Shaobkj_FreeColor": Shaobkj_FreeColor,
    "Shaobkj_Text_Process": Shaobkj_Text_Process,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Shaobkj_APINode": "🤖图像生成",
    "Shaobkj_APINode_Batch": "🤖并发-编辑-文本驱动",
    "Shaobkj_Sora_Video": "🤖 Shaobkj -Sora视频",
    "Shaobkj_Veo_Video": "🤖 Shaobkj -Veo视频",
    "Shaobkj_ConcurrentImageEdit_Sender": "🤖并发-编辑-图像驱动",
    "Shaobkj_GroupedConcurrentImageEdit": "🧩组合并发",
    "Shaobkj_Load_Image_Path": "🤖加载图像",
    "Shaobkj_Load_Batch_Images": "🤖批量加载图像(路径)",
    "Shaobkj_Image_Save": "🤖图像保存",
    "Shaobkj_Fixed_Seed": "🤖固定随机种子",
    "Shaobkj_LLM_App": "🤖LLM应用",
    "Shaobkj_ForLoop_Start": "🤖循环开始",
    "Shaobkj_ForLoop_End": "🤖循环结束",
    "Shaobkj_NanoBanana_Prompt": "🤖香蕉专属提示词",
    "Shaobkj_LoadImageListFromDir": "🤖加载图像列表(路径)",
    "Shaobkj_ResolutionJudge": "🤖分辨率智能判断",
    "Shaobkj_GetEdgeLength": "🤖获取边长",
    "Shaobkj_Load_Florence2_Model": "🤖加载Florence2模型(急速)",
    "Shaobkj_Florence2_Fast_Prompt": "🤖图像急速反推",
    "Shaobkj_SeamlessPattern": "🎨 接回头 (四方连续)",
    "Shaobkj_FreeColor": "🎨 自由调色",
    "Shaobkj_Text_Process": "📝 文本处理",
}

print("[ComfyUI-shaobkj] Node code loaded. Please restart ComfyUI if you see this message for the first time.")
