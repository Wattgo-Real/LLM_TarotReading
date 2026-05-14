import http.server
import socketserver
import os
import base64
import json
from datetime import datetime
import torch

from Inference import load_model, load_image_from_path, run_model

PORT = 8000
DIRECTORY = "captures"

device = "cuda" if torch.cuda.is_available() else "cpu"

config_path = "./RT-DETRv2-repo/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_tarot.yml"
model_path = "./output/rtdetrv2_r18vd_120e_tarot/checkpoint0016.pth"
output_dir = "./inference_result"

model = load_model(model_path, config_path, device)


if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/upload':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            image_data = data['image'].split(',')[1]
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(DIRECTORY, filename)
            
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(image_data))
            
            print(f"Saved: {filepath}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {'status': 'success', 'filename': filename}
            self.wfile.write(json.dumps(response).encode())

            image_path = filepath
            run_model(model, image_path, output_dir)
        else:
            super().do_POST()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

print(f"--- Lumina Cam Server ---")
print(f"1. 確保手機與電腦在同一個 Wi-Fi 網路")
print(f"2. 在手機瀏覽器輸入: http://{get_ip()}:{PORT}")
print(f"--------------------------")

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", PORT), UploadHandler) as httpd:
    httpd.serve_forever()
