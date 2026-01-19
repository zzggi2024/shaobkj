import websocket # NOTE: websocket-client (pip install websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import requests # NOTE: requests (pip install requests)
import os
import time

# =================================================================================
# 配置区域 (Configuration)
# =================================================================================

# 云端 ComfyUI 地址 (Cloud ComfyUI Address)
COMFYUI_SERVER_ADDRESS = "127.0.0.1:8188" # 请修改为您的云主机 IP:端口 (e.g., "192.168.1.100:8188")
CLIENT_ID = str(uuid.uuid4())

# 本地文件设置
LOCAL_IMAGE_PATH = "test_image.jpg"   # 想要处理的本地图片路径
OUTPUT_FOLDER = "local_results"       # 结果保存的本地目录

# =================================================================================
# 核心函数 (Core Functions)
# =================================================================================

def queue_prompt(prompt):
    """提交任务到 ComfyUI 队列"""
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFYUI_SERVER_ADDRESS}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    """下载生成的图片"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    """获取任务历史结果"""
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}") as response:
        return json.loads(response.read())

def upload_image(filepath):
    """上传本地图片到云端 ComfyUI"""
    if not os.path.exists(filepath):
        print(f"❌ 错误: 找不到文件 {filepath}")
        return None

    with open(filepath, 'rb') as file:
        files = {'image': file}
        data = {'overwrite': 'true'} # 覆盖同名文件
        
        print(f"📤 正在上传: {filepath} -> 云端...")
        response = requests.post(
            f"http://{COMFYUI_SERVER_ADDRESS}/upload/image", 
            files=files, 
            data=data
        )
        
    if response.status_code == 200:
        result = response.json()
        name = result.get("name")
        print(f"✅ 上传成功: {name}")
        return name
    else:
        print(f"❌ 上传失败: {response.status_code} - {response.text}")
        return None

def get_images(ws, prompt):
    """执行并等待结果"""
    prompt_id = queue_prompt(prompt)['prompt_id']
    print(f"⏳ 任务已提交 ID: {prompt_id}，等待执行...")
    
    output_images = {}
    
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    print("✅ 任务执行完成！")
                    break # Execution is done
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        if 'images' in node_output:
            images_output = []
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append({
                    "filename": image['filename'],
                    "data": image_data
                })
            output_images[node_id] = images_output

    return output_images

# =================================================================================
# 主逻辑 (Main Logic)
# =================================================================================

def run_local_bridge(workflow_api_json_path):
    # 1. 检查本地目录
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. 连接 WebSocket
    ws = websocket.WebSocket()
    try:
        ws.connect(f"ws://{COMFYUI_SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")
        print("🔗 已连接到云端 ComfyUI WebSocket")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 提示: 请确保云主机地址配置正确，且防火墙允许访问 8188 端口。")
        return

    # 3. 加载 Workflow 模板
    if not os.path.exists(workflow_api_json_path):
        print(f"❌ 错误: 找不到 workflow 文件 {workflow_api_json_path}")
        return
        
    with open(workflow_api_json_path, 'r', encoding='utf-8') as f:
        prompt_workflow = json.load(f)

    # 4. 上传图片
    uploaded_filename = upload_image(LOCAL_IMAGE_PATH)
    if not uploaded_filename:
        return

    # 5. 修改 Workflow 中的 LoadImage 节点
    # 注意: 这里需要你根据实际的 Workflow 结构来修改
    # 假设 LoadImage 节点的 ID 是 "3" (请在 workflow_api.json 中确认)
    target_node_id = None
    for node_id, node_info in prompt_workflow.items():
        if node_info["class_type"] == "LoadImage":
            target_node_id = node_id
            print(f"🔍 找到 LoadImage 节点 ID: {node_id}")
            break
    
    if target_node_id:
        prompt_workflow[target_node_id]["inputs"]["image"] = uploaded_filename
    else:
        print("⚠️ 警告: 未在 Workflow 中找到 LoadImage 节点，将直接运行原 Workflow...")

    # 6. 执行并获取结果
    print("🚀 开始云端处理...")
    try:
        images = get_images(ws, prompt_workflow)
        
        # 7. 保存结果到本地
        for node_id, image_list in images.items():
            for img in image_list:
                file_name = f"result_{int(time.time())}_{img['filename']}"
                save_path = os.path.join(OUTPUT_FOLDER, file_name)
                with open(save_path, 'wb') as f:
                    f.write(img['data'])
                print(f"💾 结果已保存至本地: {save_path}")
                
    except Exception as e:
        print(f"❌ 执行出错: {e}")
    finally:
        ws.close()

if __name__ == "__main__":
    # 使用说明
    print("--- ComfyUI 云端-本地 桥接工具 ---")
    print("请先准备好一个 'workflow_api.json' 文件 (从 ComfyUI -> 保存(API格式))")
    # run_local_bridge("workflow_api.json") 
