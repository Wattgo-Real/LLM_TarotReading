import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms

import random
import glob
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json

font = ImageFont.truetype("msjh.ttc", 24)


# Add repo to path
repo_path = os.path.abspath("./RT-DETRv2-repo/rtdetrv2_pytorch")
if repo_path not in sys.path:
    sys.path.append(repo_path)

from src.core import YAMLConfig



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def get_class_names(label_file = "./label.json"):
    with open(label_file, "r", encoding="utf-8") as f:
        label = json.load(f)
    return label

def get_results(image, labels, boxes, scores, class_names, font_size = 20, threshold=0.5):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("msjh.ttc", size = font_size)
    except:
        font = ImageFont.load_default(size = font_size)

    # Filter results by threshold
    scr = scores[0]
    lab = labels[0][scr > threshold]
    box = boxes[0][scr > threshold]
    scrs = scores[0][scr > threshold]
    get_card = []
    for i, b in enumerate(box):
        b = b.tolist()
        label_id = int(lab[i].item())
        label_name = str(label_id)
        get_card.append([b[0], label_name, scrs[i].item()])

        score = scrs[i].item()
        
        # Draw box
        draw.rectangle(b, outline="#00FF00", width=2)
        # Draw label
        text = f"{class_names[label_name]['ch-TW']} {score:.2f}"
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=font)
        
        n = 20
        bg_coords = [b[0], b[1] - text_h - n - 8, b[0] + text_w + n + 12, b[1]]
        draw.rectangle(bg_coords, fill="#00FF00")
        draw.text((b[0] + n, b[1] - text_h - n), text, fill="black", font=font)

    get_card = sorted(get_card, key=lambda x: x[0])
    return image, get_card

def draw_results_old(image, labels, boxes, scores, class_names, font_size = 20, threshold=0.5, iou_threshold=0.45):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("msjh.ttc", size = font_size)
    except:
        font = ImageFont.load_default(size = font_size)

    # Filter results by threshold
    scr = scores[0]
    lab = labels[0][scr > threshold]
    box = boxes[0][scr > threshold]
    scrs = scores[0][scr > threshold]

    # Apply NMS (Global NMS)
    keep = nms(box, scrs, iou_threshold)
    lab = lab[keep]
    box = box[keep]
    scrs = scrs[keep]

    for i, b in enumerate(box):
        b = b.tolist()
        label_id = int(lab[i].item())
        label_name = class_names[str(label_id)]["ch-TW"]
        score = scrs[i].item()
        
        # Draw box
        draw.rectangle(b, outline="#00FF00", width=2)
        # Draw label
        text = f"{label_name} {score:.2f}"
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w = right - left
            text_h = bottom - top
        except AttributeError:
            text_w, text_h = draw.textsize(text, font=font)
        
        bg_coords = [b[0], b[1] - text_h - 16, b[0] + text_w + 16, b[1]]
        draw.rectangle(bg_coords, fill="#00FF00")
        draw.text((b[0] + 8, b[1] - text_h - 8), text, fill="black", font=font)

    return image


def load_image_from_path(selected_image_path):
    filename_extension = selected_image_path.rsplit(".")[-1]
    if filename_extension in ["jpg","jpeg","png"]:
        image = Image.open(selected_image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        width, height = image.size
        if height > width:
            image = image.rotate(90, expand=True)

        return image
    elif filename_extension in ["HEIC"]:
        try:
            import pillow_heif
        except ImportError:
            print('pillow-heif not found, use "pip install pillow-heif" to install')

        heif_file = pillow_heif.read_heif(selected_image_path)
        return Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")

def load_rt_detr_model(model_path, config_path, device):
    cfg = YAMLConfig(config_path, resume=model_path)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(device)
    model.eval()
    return model

def run_rt_detr_model(model, image_path, output_dir):
    im_pil = load_image_from_path(image_path)
    output_path = os.path.join(output_dir, os.path.basename(image_path).rsplit(".",1)[0]+".png") 

    # 3. Preprocess
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose([T.Resize((640, 640)),T.ToTensor(),])
    im_data = transforms(im_pil)[None].to(device)

    # 4. Inference
    with torch.no_grad():
        output = model(im_data, orig_size)
    
    labels, boxes, scores = output

    # 5. Visualize
    class_names = get_class_names()
    result_img, get_card = get_results(im_pil, labels, boxes, scores, class_names, font_size = w//50, threshold=0.8)

    # prompt = f"""
    # 請你當一位塔羅牌解讀大師，為我解讀這張牌卡
    # 過去：{get_card[0][1]}
    # 現在：{get_card[1][1]}
    # 未來：{get_card[2][1]}

    # 我想要問：今天的運勢如何?
    # 請你先將這三張牌的牌意結合起來，做一個完整的解讀，然後給我一些建議。
    # 請用繁體中文回答。
    # """

    result_img.save(output_path)
    print(f"Result saved to: {output_path}")

    return get_card


if __name__ == "__main__":

    config_path = "./RT-DETRv2-repo/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_tarot.yml"
    #model_path = "./output/OldModel/checkpoint0006.pth"
    model_path = "./output/NewModel/checkpoint0006.pth"
    image_dir = "./Data/TestImg"
    output_dir = "./Results/inference_result"

    model = load_rt_detr_model(model_path, config_path, device)

    image_path_list = os.listdir(image_dir)
    for image_path in image_path_list:
        full_image_path = os.path.join(image_dir, image_path) 
        run_rt_detr_model(model, full_image_path, output_dir)
        
