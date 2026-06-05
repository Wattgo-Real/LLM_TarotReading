
from IPython.core import magic_arguments
import torch
print("CUDA 是否可用：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("顯示卡名稱：", torch.cuda.get_device_name(0))

import unsloth
from unsloth import FastModel
import re
import gc


# What does my career look like in the future?
# What does my love life look like in the future?


def clean_response(text: str) -> str:
    stop_signals = [r"### Instruction:", r"### Input:", r"### Response:", r"user", r"model"]
    for signal in stop_signals:
        parts = re.split(signal, text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[0]

    text = re.sub(r'<end[_\w]*>', '', text)
    text = re.sub(r'<start_of_turn>', '', text)
    for pattern in [r'\nThe answer was.*', r'\nAnswer:.*', r'\n<start_of']:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text.strip()



def analyze_tarot(model, tokenizer, cards_input: dict) -> str:
    # 1. 設定 Chat Template (以符合 Gemma 4 訓練格式)
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    SYSTEM_PROMPT = """You are an expert Tarot card analyst with deep knowledge of all 78 cards, \
    their symbolism, numerology, and elemental associations. \
    Provide thoughtful, insightful, and compassionate readings that help people \
    gain clarity and guidance.

    Strict rules:
    1. Do not use vague guesses; the meaning of the cards must be logically combined with the user's question.
    2. Structural requirements: Please first analyze the individual cards for the [Past], [Present], and [Future] separately, and finally provide a [unified comprehensive divination prediction] that connects the energies of the three to give the final answer.
    """
        

    # SYSTEM_PROMPT = """You are an expert Tarot card analyst with deep knowledge of all 78 cards, 
    # their symbolism, numerology, and elemental associations. 
    # Provide thoughtful, insightful, and compassionate readings that help people gain clarity and guidance.
    # """

    # A. 組合抽卡文字結構
    cards_text = "\n".join([
        f"- {c['position']}: {c['card']} ({c['orientation']})"
        for c in cards_input["cards"]
    ])
    
    from Gemini import translate_to_english
    question_en = translate_to_english(cards_input.get("question", ""))
    print(question_en)

    # B. 組合對話結構
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{question_en}\n\nCards drawn:\n{cards_text}"},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # C. 轉碼
    inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")

    # 切換為推論模式（Unsloth 內建優化）
    FastModel.for_inference(model) 

    # Gemma 4 的 end_of_turn id 為 3, eos id 為 1
    terminators = [3, 1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            use_cache=True,
            do_sample=False,              # 👈 設為 False 取得穩定、最優的結果
            repetition_penalty=1.2,       # 👈 微調：防重複
            eos_token_id=terminators,
            pad_token_id=1
        )
    input_length = inputs["input_ids"].shape[1]
    real_tokenizer = tokenizer.tokenizer
    response = real_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return clean_response(response)



def analyze_tarot_split(model_lora_path: str, model_base_path: str, cards_input: dict) -> str:
    from Gemini import translate_to_english
    question_en = translate_to_english(cards_input.get("question", ""))
    print(question_en)

    # 1. 載入 LoRA 模型進行單張牌意推理
    print("--- 載入 LoRA 模型進行單張牌意推理 ---")
    model_lora, tokenizer_lora = FastModel.from_pretrained(
        model_name = model_lora_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastModel.for_inference(model_lora)

    tokenizer_lora.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    SYSTEM_PROMPT_LORA = """You are an expert Tarot reader. Simply interpret the card drawn by the user and explain its meaning specifically for their question. 
    Do not analyze it as past, present, or future."""

    card_meanings = []
    for c in cards_input["cards"]:
        single_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LORA},
            {"role": "user",   "content": f"Question: {question_en}\n\nCard drawn:\n- {c['card']} ({c['orientation']})"},
        ]
        single_prompt = tokenizer_lora.apply_chat_template(
            single_messages, tokenize=False, add_generation_prompt=True
        )
        inputs_single = tokenizer_lora(text=single_prompt, return_tensors="pt").to("cuda")
        input_length_single = inputs_single["input_ids"].shape[1]

        with torch.no_grad():
            outputs_single = model_lora.generate(
                **inputs_single,
                max_new_tokens=200,
                use_cache=True,
                do_sample=False,
                repetition_penalty=1.2,
                eos_token_id=[3, 1],
                pad_token_id=1
            )
        real_tok_lora = tokenizer_lora.tokenizer
        meaning_text = real_tok_lora.decode(outputs_single[0][input_length_single:], skip_special_tokens=True).strip()
        meaning_text = clean_response(meaning_text)
        card_meanings.append({
            "position": c["position"],
            "card": c["card"],
            "orientation": c["orientation"],
            "meaning": meaning_text
        })


    # 清理 LoRA 模型以釋放 GPU 記憶體
    print("--- 釋放 LoRA 模型 GPU 記憶體 ---")
    del model_lora
    del tokenizer_lora
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 載入 Base 模型進行綜合占卜預測
    print("--- 載入 Base 模型進行綜合占卜預測 ---")
    model_base, tokenizer_base = FastModel.from_pretrained(
        model_name = model_base_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    FastModel.for_inference(model_base)

    tokenizer_base.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
            "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    SYSTEM_PROMPT_BASE = """You are an expert Tarot card analyst with deep knowledge of all 78 cards, 
their symbolism, numerology, and elemental associations. 
Provide thoughtful, insightful, and compassionate readings that help people 
gain clarity and guidance.

Strict rules:
1. Do not use vague guesses; the meaning of the cards must be logically combined with the user's question.
2. Structural requirements: Please first analyze the individual cards for the [Past], [Present], and [Future] separately, and finally provide a [unified comprehensive divination prediction] that connects the energies of the three to give the final answer."""

    cards_inferred_text = "\n".join([
        f"- {m['position']}: {m['card']} ({m['orientation']})\n  Inferred Meaning: {m['meaning']}"
        for m in card_meanings
    ])

    messages_base = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user",   "content": f"{question_en}\n\nCards drawn and their meanings:\n{cards_inferred_text}"},
    ]

    prompt_base = tokenizer_base.apply_chat_template(
        messages_base, tokenize=False, add_generation_prompt=True
    )
    inputs_base = tokenizer_base(text=prompt_base, return_tensors="pt").to("cuda")
    input_length_base = inputs_base["input_ids"].shape[1]

    with torch.no_grad():
        outputs_base = model_base.generate(
            **inputs_base,
            max_new_tokens=400,
            use_cache=True,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=[3, 1],
            pad_token_id=1
        )
    real_tok_base = tokenizer_base.tokenizer
    response_text = real_tok_base.decode(outputs_base[0][input_length_base:], skip_special_tokens=True).strip()
    response_text = clean_response(response_text)
    print(response_text)

    # 清理 Base 模型以釋放 GPU 記憶體
    print("--- 釋放 Base 模型 GPU 記憶體 ---")
    del model_base
    del tokenizer_base
    gc.collect()
    torch.cuda.empty_cache()

    meanings_formatted = "\n\n".join([
        f"### {m['position']}: {m['card']} ({m['orientation']})\n{m['meaning']}"
        for m in card_meanings
    ])
    return meanings_formatted + "\n\n=== 綜合占卜預測 ===\n\n" + response_text



# ==========================================
# 4. 測試執行
# ==========================================
if __name__ == "__main__":


    # # ==========================================
    # # 1. 設定本地端資料夾路徑
    # # ==========================================
    # BASE_MODEL_PATH = "C:/Users/m0987/Desktop/my_local_model"

    # print("正在透過 Unsloth 專屬通道載入模型...")
    # model, tokenizer = FastModel.from_pretrained(
    #     model_name = BASE_MODEL_PATH,
    #     max_seq_length = 2048,
    #     load_in_4bit = True,
    #     device_map = {"": 0},  # 👈 霸王條款：嚴格禁止丟到 CPU，全部給我塞進 GPU！
    # )

    # print("✅ 模型載入完成")
    # print("模型類型：", type(model))



    # # ==========================================
    # # 2. 聊天範本與預設提示詞（保持你的結構）
    # # ==========================================
    # tokenizer.chat_template = (
    #     "{% for message in messages %}"
    #         "{% if message['role'] == 'system' %}<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
    #         "{% elif message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    #         "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    #         "{% endif %}"
    #     "{% endfor %}"
    #     "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    # )


    # prompt = {
    #     "question": "What does my career future look like?",
    #     "cards": [
    #         {"position": "Past",    "card": "The Fool",  "orientation": "upright"},
    #         {"position": "Present", "card": "The Tower", "orientation": "reversed"},
    #         {"position": "Future",  "card": "The Star",  "orientation": "upright"},
    #     ]
    # }
    # result = analyze_tarot(model, tokenizer, prompt)

    # print("\n--- 占卜結果 ---")
    # print(result)
    pass