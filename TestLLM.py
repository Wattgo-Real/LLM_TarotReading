


import torch
import json
from Inference import load_model, load_image_from_path, run_model
from Gemini import LLM_run

device = "cuda" if torch.cuda.is_available() else "cpu"

config_path = "./RT-DETRv2-repo/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_tarot.yml"
model_path = "./output/rtdetrv2_r18vd_120e_tarot/RT-DETRv2_for_tarot.pth"
output_dir = "./inference_result"
filepath = "./captures/capture_20260513_104112.png"

label_file = "./label.json"
with open(label_file, "r", encoding="utf-8") as f:
    label = json.load(f)


model = load_model(model_path, config_path, device)


get_cards = run_model(model, filepath, output_dir)
card_info = []
for i, card in enumerate(get_cards):
    if i == 2:
        continue

    card_info.append({"name":label[card[1]]["ch-TW"],
                      "meaning":label[card[1]].get('meaning', 'null')})


user_input = "今天運勢如何"
LLM_run(card_info, user_input)

