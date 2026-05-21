import json
import os
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv



load_dotenv()  # 自動讀取 .env 檔


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


# ==========================================
# (預留介面) 步驟 X: 影像辨識結果轉換函式
# ==========================================
# 當 CV 模組完成後，取消下方註解並修改 mock_cv_output 的輸入來源
#
# def process_cv_output(cv_result: str, tarot_db: dict) -> dict:
#     """
#     接收影像辨識模組輸出的牌名字串，查詢資料庫後回傳牌義資訊。
#     cv_result 預期格式範例: "The Fool_upright" 或 "The Magician_reversed"
#     """
#     try:
#         parts = cv_result.strip().split("_")
#         card_name = parts[0]
#         orientation = parts[1] if len(parts) > 1 else "upright"
#         return draw_card(tarot_db, force_card=card_name, force_orientation=orientation)
#     except Exception as e:
#         print(f"[CV解析錯誤] {e}，將改為隨機抽牌。")
#         return draw_card(tarot_db)


# ==========================================
# 步驟 3: 初始化 Gemini Chat 並執行對話迴圈
# ==========================================
def LLM_run(card_info, user_input):
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

    print("\n--- 模擬抽牌結果 ---")
    print(f"系統已從資料庫擷取以下背景知識：\n{retrieved_context}")

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

    print("\n初始化塔羅 AI 占卜師中...")
    chat = client.chats.create(model="gemini-3.5-flash", config=config)

    print("\n=====================================================")
    print(f"占卜師：你好！你抽到了「{card_info[0]['name']}」「{card_info[1]['name']}」「{card_info[2]['name']}")
    print(f"你今天想詢問: {user_input}")
    print("=====================================================")

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
            print(f"\n占卜師: {response.text}")
            break
        except Exception as e:
            print(f"\n[API 錯誤] {e}")
            print("請檢查你的 API Key 與網路連線。")

