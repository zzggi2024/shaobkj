import http.server
import socketserver
import os
import cgi
import json
import argparse
import socket

# =================================================================================
# 配置 (Configuration)
# =================================================================================
DEFAULT_PORT = 8000
UPLOAD_DIR = "uploads"  # 接收云端回传图片的目录

# =================================================================================
# 服务端逻辑 (Server Logic)
# =================================================================================

class LocalFileHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """处理文件读取请求"""
        # 允许跨域
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        # 解析路径参数 ?path=...
        if '?' in self.path:
            path_part, query_part = self.path.split('?', 1)
            params = dict(qc.split("=") for qc in query_part.split("&") if "=" in qc)
            
            target_path = params.get("path")
            if target_path:
                # 解码 URL 编码的路径 (e.g. %20 -> space)
                target_path = urllib.parse.unquote(target_path)
                
                if os.path.exists(target_path) and os.path.isfile(target_path):
                    print(f"📖 读取本地文件: {target_path}")
                    with open(target_path, 'rb') as f:
                        self.wfile.write(f.read())
                    return
                else:
                    print(f"❌ 文件不存在: {target_path}")
                    self.wfile.write(b"File not found")
                    return
        
        # 默认行为：显示当前目录
        super().do_GET()

    def do_POST(self):
        """处理文件上传请求"""
        # 允许跨域
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if self.path == '/upload':
            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
            if ctype == 'multipart/form-data':
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                fields = cgi.parse_multipart(self.rfile, pdict)
                
                # 获取文件数据
                file_data = fields.get('file')
                filename = fields.get('filename')
                
                if file_data and filename:
                    # 确保是列表中的第一个（cgi返回的是列表）
                    data = file_data[0] if isinstance(file_data, list) else file_data
                    fname = filename[0] if isinstance(filename, list) else filename
                    
                    # 确保上传目录存在
                    if not os.path.exists(UPLOAD_DIR):
                        os.makedirs(UPLOAD_DIR)
                        
                    save_path = os.path.join(UPLOAD_DIR, fname)
                    with open(save_path, 'wb') as f:
                        f.write(data)
                        
                    print(f"💾 已保存回传文件: {save_path}")
                    response = {"status": "success", "path": os.path.abspath(save_path)}
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.wfile.write(b'{"status": "error", "message": "No file data"}')
            else:
                self.wfile.write(b'{"status": "error", "message": "Content-Type must be multipart/form-data"}')
        else:
            self.wfile.write(b'{"status": "error", "message": "Invalid endpoint"}')

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    import urllib.parse
    
    parser = argparse.ArgumentParser(description="ComfyUI 本地文件桥接服务")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="服务端口")
    args = parser.parse_args()

    # 切换到脚本所在目录，确保 uploads 文件夹创建在正确位置
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n--- ComfyUI 本地文件桥接服务 ---")
    print(f"✅ 服务已启动，监听端口: {args.port}")
    print(f"📂 回传文件将保存在: {os.path.abspath(UPLOAD_DIR)}")
    print(f"\n⚠️  注意：云主机必须能访问到本机 IP！")
    print(f"   本机局域网 IP: {get_local_ip()}")
    print(f"   如果云主机在公网，请使用内网穿透工具 (如 cpolar/ngrok) 将本机 {args.port} 端口暴露到公网。")
    print(f"   示例公网地址: http://xxxx.cpolar.cn")
    print(f"\n--- 日志 ---")

    with socketserver.TCPServer(("", args.port), LocalFileHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 服务已停止")
