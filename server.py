import http.server
import socketserver
import os
import base64
import json
from datetime import datetime
import torch
from Gemini import LLM_run, translate_to_chinese
from gamma_4_with_lora import analyze_tarot as Self_LLM_run, analyze_tarot_split, clean_response

from Inference import load_rt_detr_model, run_rt_detr_model



GAMMA_MODEL_PATH = "C:/Users/User/Desktop/Model/gemma_model"
GAMMA_MODEL_LORA_PATH = "C:/Users/User/Desktop/Model/gemma_model_with_lora"


# 讀取 Gamma 4 LLM+LoRA 模型
def Load_LLM_Model(lora = True):
    from unsloth import FastModel
    model_path = GAMMA_MODEL_LORA_PATH if lora else GAMMA_MODEL_PATH

    print("正在透過 Unsloth 專屬通道載入模型...")
    model_LLM, tokenizer = FastModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
        #device_map = {"": 0},  # 選擇 GPU
    )
    FastModel.for_inference(model_LLM)

    print(f"✅ 模型載入完成，模型類型：{type(model_LLM)}")
    return model_LLM, tokenizer





PORT = 8000

device = "cuda" if torch.cuda.is_available() else "cpu"

config_path = "./RT-DETRv2-repo/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_tarot.yml"
model_path = "./output/NewModel/checkpoint0006.pth"
output_dir = "./Results/inference_result"
DIRECTORY = "./Results/captures"
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

model_detr = load_rt_detr_model(model_path, config_path, device)





class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        print(f"--- OPTIONS request: {self.path} ---")
        self.send_response(200)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        parsed_path = self.path.split('?')[0]
        print(f"--- POST request: {self.path} (parsed: {parsed_path}) ---")
        if parsed_path == '/upload':
            print("a")
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            image_data = data['image'].split(',')[1]
            self.server.time_now = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"capture_{self.server.time_now}.png"
            filepath = os.path.join(DIRECTORY, filename)
            
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(image_data))
            
            print(f"Saved: {filepath}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'filename': filename}
            self.wfile.write(json.dumps(response).encode())

            image_path = filepath
            get_cards = run_rt_detr_model(model_detr, filepath, output_dir)

            label_file = "./label.json"
            with open(label_file, "r", encoding="utf-8") as f:
                label = json.load(f)


            card_info = []
            for i, card in enumerate(get_cards):
                card_info.append({"name":label[card[1]]["en"],       # en, ch-TW
                                "meaning":label[card[1]].get('meaning', 'null')})
            
            self.server.card_info = card_info
            self.check_and_run()

        elif parsed_path == '/question':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            question = data.get('question', '')
            self.server.question = question
            #with open("question.txt", "w", encoding="utf-8") as f:
            #    f.write(question)
                
            print(f"Saved question: question.txt")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success'}
            self.wfile.write(json.dumps(response).encode())
            self.check_and_run()

        elif parsed_path == '/settings':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            self.server.use_self_model = bool(data.get('use_self_model', False))
            self.server.use_split_inference = bool(data.get('use_split_inference', False))
            print(f"Settings updated: use_self_model={self.server.use_self_model}, use_split_inference={self.server.use_split_inference}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            response = {'status': 'success'}
            self.wfile.write(json.dumps(response).encode())

        else:
            super().do_POST()

    def do_GET(self):
        parsed_path = self.path.split('?')[0]
        print(f"--- GET request: {self.path} (parsed: {parsed_path}) ---")
        if parsed_path == '/result':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            time_now = getattr(self.server, "time_now", "")
            text_path = f"./Results/LLM_result/result_{time_now}.txt"
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    content = f.read()
                response = {'status': 'success', 'data': content}
            else:
                response = {'status': 'pending'}
                
            self.wfile.write(json.dumps(response).encode())
            
        elif parsed_path == '/settings':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            response = {
                'use_self_model': getattr(self.server, 'use_self_model', False),
                'use_split_inference': getattr(self.server, 'use_split_inference', True)
            }
            self.wfile.write(json.dumps(response).encode())
            
        else:
            super().do_GET()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def check_and_run(self):
        s = getattr(self.server, "card_info", None)
        p = getattr(self.server, "question", None)
        print(s, p)
        time_now = getattr(self.server, "time_now", "")
        if s is not None and p is not None:
            self.server.card_info = None
            self.server.question = None
            
            cards_get = []
            time_get = ["Past", "Present", "Future"]
            for i in range(3):
                if "Reversed" in s[i]['name']:
                    type = "reversed"
                else: 
                    type = "upright"
                cards_get.append({"position": time_get[i], "card": s[i]['name'].replace(" Reversed",""), "orientation": type})
            prompt = {"question": p, "cards": cards_get}
            print("輸入資訊:", prompt)

            use_self_model_val = getattr(self.server, 'use_self_model', False)
            use_split_inference_val = getattr(self.server, 'use_split_inference', True)

            if use_self_model_val == False:
                print("使用 gemini 3.5 flash 進行預測")
                response_text = LLM_run(prompt)
            else:
                print("使用 Gemma 4 (LORA) 進行本地端預測")
                if use_split_inference_val:
                    response_text = analyze_tarot_split(
                        GAMMA_MODEL_LORA_PATH,
                        GAMMA_MODEL_PATH,
                        prompt
                    )
                else:
                    if self.server.model_LLM_lora is None or self.server.tokenizer_lora is None:
                        self.server.model_LLM_lora, self.server.tokenizer_lora = Load_LLM_Model(lora = True)
                    response_text = Self_LLM_run(self.server.model_LLM_lora, self.server.tokenizer_lora, prompt)
                
                print("將本地端推論結果翻譯成中文...")
                response_text = translate_to_chinese(response_text)

            print(response_text)
            save_path = f"./Results/LLM_result/result_{time_now}.txt"
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
                pass

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
    httpd.card_info = None
    httpd.question = None
    httpd.model_LLM_lora = None
    httpd.tokenizer_lora = None
    httpd.model_LLM_base = None
    httpd.tokenizer_base = None
    httpd.use_self_model = False
    httpd.use_split_inference = True

    httpd.serve_forever()
