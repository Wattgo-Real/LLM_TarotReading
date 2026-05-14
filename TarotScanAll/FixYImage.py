import cv2
import numpy as np

def fix_yellowish_image(image_path, output_path):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print("無法讀取圖片，請檢查路徑。")
        return

    # 將 BGR 轉換為 LAB 色彩空間
    # LAB 空間將亮度 (L) 與色彩資訊 (A, B) 分開，更容易處理色偏
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 實作簡單的白平衡：將 A 和 B 通道（負責顏色）的均值移回中心點 (128)
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    a = cv2.addWeighted(a, 1, np.zeros_like(a), 0, 128 - a_mean)
    b = cv2.addWeighted(b, 1, np.zeros_like(b), 0, 128 - b_mean)

    # 合併通道並轉回 BGR
    fixed_lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(fixed_lab, cv2.COLOR_LAB2BGR)

    # (可選) 增加一點對比度讓文字更清晰
    # 我們可以稍微拉伸亮度通道 (L)
    # result = cv2.convertScaleAbs(result, alpha=1.1, beta=0)

    cv2.imwrite(output_path, result)
    print(f"修正完成！已儲存至: {output_path}")

# 使用範例
for i in range(1,10):
    fix_yellowish_image(f'./TarotScan/{i}.jpg', f'./TarotScan/{i}_r.jpg')