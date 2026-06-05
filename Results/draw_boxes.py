import cv2
import json
import os

# 檔案路徑設定
image_path = r"C:\Users\User\Desktop\TarotTrainingData\new_dataset_train\enhanced_116.jpg"
label_path = r"C:\Users\User\Desktop\TarotTrainingData\new_dataset_train\enhanced_116.txt"
label_json_path = "./label.json"  # 專案中的標籤對照表
output_path = r"./Results/enhanced_116_boxed.jpg"

def main():
    # 1. 讀取影像
    if not os.path.exists(image_path):
        print(f"找不到影像檔案: {image_path}")
        return
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # 2. 讀取標籤對照表 (label.json)
    if os.path.exists(label_json_path):
        with open(label_json_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        print(f"警告: 找不到 {label_json_path}，將直接顯示類別 ID。")
        label_map = {}

    # 3. 讀取 YOLO 標籤檔案
    if not os.path.exists(label_path):
        print(f"找不到標籤檔案: {label_path}")
        return

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])

        # 4. 將 YOLO 標準化座標 (0~1) 轉換為影像像素座標
        x_min = int((x_center - bbox_width / 2) * w)
        y_min = int((y_center - bbox_height / 2) * h)
        x_max = int((x_center + bbox_width / 2) * w)
        y_max = int((y_center + bbox_height / 2) * h)

        # 5. 取得對應卡牌名稱
        card_name = class_id
        if class_id in label_map:
            # 優先使用中文，也可改為 .get("en") 使用英文
            card_name = label_map[class_id].get("ch-TW", label_map[class_id].get("en", class_id))

        # 6. 畫框與文字
        # 畫綠框 (BGR: 0, 255, 0)，粗細度設為 2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 繪製文字背景與標籤
        text = f"{card_name} (ID:{class_id})"
        # 由於 OpenCV 內建的 putText 不支援中文，若要顯示中文需要 Pillow 轉換或只顯示英文。
        # 這裡提供英文標籤 fallback 以免中文亂碼，或使用 OpenCV 的英文顯示
        if class_id in label_map:
            english_name = label_map[class_id].get("en", class_id)
            text = f"{english_name} (ID:{class_id})"

        # 畫文字底色背景以利閱讀
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x_min, y_min - text_h - 10), (x_min + text_w, y_min), (0, 255, 0), -1)
        cv2.putText(img, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # 7. 儲存結果
    cv2.imwrite(output_path, img)
    print(f"成功將標註框框畫出，並儲存至: {output_path}")

if __name__ == "__main__":
    main()
