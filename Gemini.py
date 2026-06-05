import json
import os
import random
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv



load_dotenv()  # 自動讀取 .env 檔


def translate_to_english(text: str, method: str = "google") -> str:
    if method == "google":
        import urllib.request
        import urllib.parse
        import json
        try:
            url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(text)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                data = json.loads(html)
                translated = "".join([sentence[0] for sentence in data[0]])
                return translated.strip()
        except Exception as e:
            print(f"[Google Translate Error] {e}")
            method = "llm"  # Fallback to LLM

    if method == "llm":
        if not os.environ.get("GEMINI_API_KEY"):
            return text
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=f"Translate the following user question into clear English for a Tarot reading context. Output ONLY the English translation, no other text or explanation:\n{text}"
            )
            return response.text.strip()
        except Exception as e:
            print(f"[LLM Translation Error] {e}")
            return text
    return text


def translate_to_chinese(text: str, method: str = "google") -> str:
    if method == "google":
        import urllib.request
        import urllib.parse
        import json
        try:
            url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=zh-TW&dt=t&q=" + urllib.parse.quote(text)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                data = json.loads(html)
                translated = "".join([sentence[0] for sentence in data[0]])
                return translated.strip()
        except Exception as e:
            print(f"[Google Translate to Chinese Error] {e}")
            method = "llm"  # Fallback to LLM

    if method == "llm":
        if not os.environ.get("GEMINI_API_KEY"):
            return text
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=f"Translate the following English Tarot reading analysis into natural, fluent, and compassionate Traditional Chinese (繁體中文). Output ONLY the translated text, preserving markdown structure if any, no other explanation:\n{text}"
            )
            return response.text.strip()
        except Exception as e:
            print(f"[LLM Translation to Chinese Error] {e}")
            return text
    return text





# set GEMINI_API_KEY= "AIzaSyD7-8yYuZORNsem3SYz21YUA2c8L8f6xss"
# ==========================================
# 步驟 1: 載入本地端的塔羅牌義資料庫
# ==========================================
def load_tarot_db(filepath="tarot_db.json"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[錯誤] 找不到 {filepath}！請確保檔案與本程式在同一目錄。")
        return {}


# ==========================================
# 步驟 2: 抽牌邏輯
# ==========================================
def draw_card(tarot_db, force_card=None, force_orientation=None):
    """
    從資料庫中抽一張牌。
    - force_card: 指定牌名（測試用），None 則隨機抽
    - force_orientation: 指定 'upright' 或 'reversed'，None 則隨機
    回傳 dict: { card_name, orientation, orientation_zh, meaning }
    """
    card_name = force_card if force_card else random.choice(list(tarot_db.keys()))

    if card_name not in tarot_db:
        print(f"[警告] 牌名 '{card_name}' 不在資料庫中，改為隨機抽牌。")
        card_name = random.choice(list(tarot_db.keys()))

    orientation = force_orientation if force_orientation else random.choice(["upright", "reversed"])
    orientation_zh = "正位" if orientation == "upright" else "逆位"
    meaning = tarot_db[card_name].get(orientation, "無此方位解釋")

    return {
        "card_name": card_name,
        "orientation": orientation,
        "orientation_zh": orientation_zh,
        "meaning": meaning,
    }


def LLM_run(prompt_get):
    """
    使用 Gemini API (gemini-3.5-flash) 進行塔羅占卜
    """

    # --- 檢查 API Key ---
    if not os.environ.get("GEMINI_API_KEY"):
        print("[錯誤] 請先設定環境變數 GEMINI_API_KEY。")
        return None

    # --- 建立 Gemini Client ---
    client = genai.Client()

    # --- 重構：融合 Alpaca 與心理諮商邏輯的 System Instruction ---
    system_instruction = (
        "You are an expert Tarot card analyst with deep knowledge of all 78 cards, their symbolism, numerology, and elemental associations. Provide thoughtful, insightful, and compassionate readings that help people gain clarity and guidance.\n"
        "Strict rules:\n"
        "1. Do not use vague guesses; the meaning of the cards must be logically combined with the user's question.\n"
        "2. Structural requirements: Please first analyze the individual cards for the [Past], [Present], and [Future] separately, and finally provide a [unified comprehensive divination prediction] that connects the energies of the three to give the final answer.\n"
        "3. Please respond in Traditional Chinese.\n"
    )

    # --- 將資料庫的牌義組合成結構化的 Input 內容 ---
    cards_text = "\n".join([
        f"- {c['position']}: {c['card']} ({c['orientation']})"
        for c in prompt_get["cards"]
    ])

    # --- 模擬 Alpaca 的 Input 結構，打包傳給 Gemini ---
    # 這樣寫可以讓 Gemini 非常精準地把「問題」跟「牌義背景」分開處理
    prompt = (
        f"### I drew: {cards_text}\n\n"
        f"### User Question: {prompt_get['question']}\n\n"
        f"### Response:"
    )

    # --- 設定 Config ---
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.5, # 保持 0.5 的溫度，讓解牌既有邏輯又富有文采
    )

    while 1:
        try:
            # 使用最新的聊天或直接生成模式（這裡用單次生成來跟前面的本地端做對比）
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=config
            )
            
            response_text = f"{response.text}"
            return response_text
            
        except Exception as e:
            print(f"\n[API 錯誤] {e}")
            print("請檢查你的 API Key 與網路連線。")
            return None
        time.sleep(1)

# ==========================================
# 步驟 3: 初始化 Gemini Chat 並執行對話迴圈
# ==========================================
def LLM_run_old(prompt):

    ALPACA_PROMPT_WITH_DB = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    你是一位專業、具備同理心且嚴謹的塔羅牌解讀師。
    請遵循以下嚴格規則：
    1. 不使用模糊猜測，必須將牌義與使用者問題進行邏輯結合。
    2. 若使用者詢問塔羅占卜以外的問題（如程式、數學等），請禮貌拒絕並引導回占卜主題。

    ### Input:
    【使用者問題】：
    {question}

    【抽出的塔羅牌與資料庫標準牌義】：
    {retrieved_context}

    ### Response:
    """
    
    # --- 載入資料庫 ---
    tarot_db = load_tarot_db()
    if not tarot_db:
        return

    # --- 檢查 API Key ---
    if not os.environ.get("GEMINI_API_KEY"):
        print("[錯誤] 請先設定環境變數 GEMINI_API_KEY。")
        print("  Linux/Mac: export GEMINI_API_KEY='你的金鑰'")
        print("  Windows:   set GEMINI_API_KEY=你的金鑰")
        return

    retrieved_context = (
        f"【抽出的過去牌】: {card_info[0]['name']} "
        f"標準牌義: {card_info[0]['meaning']}"
        f"【抽出的現在牌】: {card_info[1]['name']} "
        f"標準牌義: {card_info[1]['meaning']}"
        f"【抽出的未來牌】: {card_info[2]['name']} "
        f"標準牌義: {card_info[2]['meaning']}"
    )

    # --- 建立 Gemini Client ---
    client = genai.Client()

    system_instruction = (
        "你是一位專業、具備同理心且嚴謹的塔羅牌解讀師，當標準牌義為 null 時請自行分析牌義。\n"
        "【嚴格規則】：\n"
        "1. 你『只能』根據我提供的『標準牌義』進行解讀，絕對不能自行發明卡牌意涵。\n"
        "2. 不使用模糊猜測，必須將牌義與使用者問題進行邏輯結合。\n"
        "3. 若使用者詢問塔羅占卜以外的問題（如程式、數學等），請禮貌拒絕並引導回占卜主題。\n"
        "4. 回應請帶有心理諮商的溫暖語氣，提供具建設性的建議，回應使用正體中文。"
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3,
    )

    #print("\n初始化塔羅 AI 占卜師中...")
    chat = client.chats.create(model="gemini-3.5-flash", config=config)

    #print("\n=====================================================")
    #print(f"占卜師：你好！你抽到了「{card_info[0]['name']}」「{card_info[1]['name']}」「{card_info[2]['name']}")
    #print(f"你今天想詢問: {user_input}")
    #print("=====================================================")

    is_first_turn = True

    while True:
        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n占卜師：感謝你今天的分享，祝你一切順心。再見！")
            break

        if is_first_turn:
            # 第一輪：將牌義作為 RAG 背景知識注入 prompt
            prompt = (
                f"以下是使用者抽出的塔羅牌與對應的標準牌義：\n"
                f"{retrieved_context}\n\n"
                f"使用者的問題是：「{user_input}」\n"
                f"請嚴格基於上述牌義，為使用者進行詳細且溫暖的解讀。"
            )
            is_first_turn = False
        else:
            # 後續輪：Chat Session 已記住牌義與解讀歷史，直接傳入使用者訊息
            prompt = user_input


        try:
            response = chat.send_message(prompt)
            response_text = f"\n占卜師: {response.text}"

            return response_text
        
        except Exception as e:
            print(f"\n[API 錯誤] {e}")
            print("請檢查你的 API Key 與網路連線。")

