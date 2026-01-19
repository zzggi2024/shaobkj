import requests
import torch
import numpy as np
from PIL import Image, ImageOps
import io
import time
import os
import folder_paths
import json

class Shaobkj_HTTP_Load_Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "local_server_url": ("STRING", {"default": "http://localhost:8000", "multiline": False, "tooltip": "本地服务地址，如 http://192.168.1.5:8000 或内网穿透地址"}),
                "file_path": ("STRING", {"default": "test.jpg", "multiline": False, "tooltip": "本地文件的绝对路径或相对路径"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_url"
    CATEGORY = "🤖shaobkj-APIbox/LocalBridge"

    def load_image_from_url(self, local_server_url, file_path):
        # 1. 基础校验：检查 URL 是否填写
        if not local_server_url or local_server_url.strip() == "":
            raise RuntimeError("❌ 错误：请填写 [本地服务地址] (local_server_url)。\n💡 提示：请在本地运行 tools/local_http_server.py，并填入显示的地址（如 http://192.168.x.x:8000）。")

        # 2. 基础校验：检查文件路径是否填写
        if not file_path or file_path.strip() == "":
            raise RuntimeError("❌ 错误：请填写 [本地文件路径] (file_path)。\n💡 提示：请填入您电脑上图片的完整路径（例如 D:\\photos\\img.jpg）。")

        # 构造请求 URL
        # 对应 tools/local_http_server.py 的 GET 处理逻辑
        base = local_server_url.rstrip("/")
        # URL encode path
        import urllib.parse
        encoded_path = urllib.parse.quote(file_path)
        url = f"{base}/?path={encoded_path}"

        print(f"[Shaobkj-HTTP] Loading from: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            
            # 3. 连接失败处理 (无法连接到本地服务)
            # requests.get 抛出 ConnectionError 会被下面的 except 捕获，这里主要处理 HTTP 状态码错误
            if response.status_code != 200:
                if response.status_code == 404:
                     raise RuntimeError(f"❌ 错误：本地文件未找到。\n💡 提示：请检查路径是否正确：{file_path}\n以及确保 tools/local_http_server.py 正在您的电脑上运行。")
                else:
                     raise RuntimeError(f"❌ 错误：服务器返回状态码 {response.status_code}。\n详细信息：{response.text}")
            
            # 4. 成功获取数据，尝试解析图片
            img_bytes = response.content
            try:
                img = Image.open(io.BytesIO(img_bytes))
            except Exception:
                raise RuntimeError(f"❌ 错误：下载的文件不是有效的图片格式。\n💡 提示：请确认 {file_path} 是一个图片文件。")

            # Standard ComfyUI image processing
            img = ImageOps.exif_transpose(img)
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                
            return (image, mask)
            
        except requests.exceptions.ConnectionError:
             raise RuntimeError(f"❌ 错误：无法连接到本地服务 ({base})。\n💡 提示：\n1. 请确保您已在本地电脑运行了 tools/local_http_server.py。\n2. 如果云端在公网，请确保您使用了内网穿透地址（如 http://xxxx.cpolar.cn），而不是局域网 IP。")
        except Exception as e:
            # 如果已经是我们自定义的 RuntimeError，直接抛出
            if "❌" in str(e):
                raise e
            print(f"[Shaobkj-HTTP] Error: {e}")
            raise RuntimeError(f"❌ 未知错误：{str(e)}")


class Shaobkj_HTTP_Send_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "local_server_url": ("STRING", {"default": "http://localhost:8000", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_Result", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "send_image_to_url"
    OUTPUT_NODE = True
    CATEGORY = "🤖shaobkj-APIbox/LocalBridge"

    def send_image_to_url(self, images, local_server_url, filename_prefix="ComfyUI_Result"):
        results = []
        
        # Ensure URL ends with /upload
        base = local_server_url.rstrip("/")
        if not base.endswith("/upload"):
            url = f"{base}/upload"
        else:
            url = base

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Create a filename
            filename = f"{filename_prefix}_{int(time.time())}_{batch_number}.png"
            
            # Convert to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            
            # Send POST request
            print(f"[Shaobkj-HTTP] Sending to: {url}")
            try:
                files = {'file': (filename, img_bytes, 'image/png'), 'filename': (None, filename)}
                response = requests.post(url, files=files, timeout=60)
                
                if response.status_code == 200:
                    print(f"[Shaobkj-HTTP] Success: {filename}")
                    results.append(f"Success: {filename}")
                else:
                    print(f"[Shaobkj-HTTP] Failed: {response.status_code}")
                    results.append(f"Failed: {response.status_code}")
                    raise RuntimeError(f"❌ 错误：上传回本地失败 (状态码 {response.status_code})。\n💡 提示：请检查本地服务 tools/local_http_server.py 是否正常运行。")
                    
            except requests.exceptions.ConnectionError:
                 raise RuntimeError(f"❌ 错误：无法连接到本地服务进行回传。\n💡 提示：\n1. 请确保您已在本地电脑运行了 tools/local_http_server.py。\n2. 检查内网穿透连接是否断开。")
            except Exception as e:
                # 如果已经是我们自定义的 RuntimeError，直接抛出
                if "❌" in str(e):
                    raise e
                print(f"[Shaobkj-HTTP] Error sending {filename}: {e}")
                results.append(f"Error: {str(e)}")
                raise RuntimeError(f"❌ 回传文件时发生未知错误: {str(e)}")

        return (", ".join(results),)
