
import json
import cv2
import os
import numpy as np


# d = {f"{i}":1 for i in range(78)}
# with open("cv.json", mode = "w", encoding="utf-8") as json_f:
#     json.dump(d, json_f, ensure_ascii=False)  # ensure_ascii 確保編碼成 utf-8 時不會出錯
CANNY_LOW = 0
CANNY_HIGH = 100

VIDEO_PATH = 'C:/Users/User/Desktop/TarotCardsVideo.mp4'  # 你的影片路徑
OUTPUT_DIR = 'TarotImgReal'
FRAME_INTERVAL = 10                 # 每 30 幀處理一次
MIN_AREA = 180*320/2                    # 最小牌面面積 (根據解析度調整)
MAX_AREA = 250*400                   # 最大牌面面積
ASPECT_RATIO_RANGE = (2.0, 1.2)     # 塔羅牌常見的長寬比 (高/寬)

# 塔羅牌長寬比 (高/寬)
TARGET_ASPECT_RATIO = 1.6
ASPECT_RATIO_TOLERANCE = 0.2 # 允許誤差 +-0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)



def process_video_visual(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤：無法打開影片文件 {video_path}")
        return

    frame_count = 0

    print("開始處理。在顯示視窗中按 'q' 退出。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue

        # 為了流暢感，每一幀都顯示，但只有特定幀才更新偵測結果
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_cards = []
        cards_center = []
        if hierarchy is not None:
            hierarchy = hierarchy[0] # 簡化結構

            for i, cnt in enumerate(contours):
                # A. 基礎篩選：面積與幾何形狀
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # 必須是凸四邊形 (近似矩形)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                    
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                aspect_ratio = h / float(w)

                if not (MIN_AREA < area < MAX_AREA):
                    continue
                    
                if not (TARGET_ASPECT_RATIO - ASPECT_RATIO_TOLERANCE < aspect_ratio < TARGET_ASPECT_RATIO + ASPECT_RATIO_TOLERANCE):
                    continue

                # B. 進階階層篩選 (尋找「黑白黑」結構)
                # hierarchy[i] 結構: [Next, Previous, First_Child, Parent]
                child_idx = hierarchy[i][2]   # 這個輪廓有沒有子輪廓
                parent_idx = hierarchy[i][3]  # 這個輪廓有沒有父輪廓

                # 在 image_1.png 的結構中：
                # 我們尋找的是「內細亮框」或「內黑框起始線」。
                # 它們通常具有較深的階層關係。
                # 這裡我們嘗試尋找「同時有父輪廓與子輪廓」的矩形，
                # 這通常代表它處於 雙框結構 的中間層。

                if parent_idx != -1 and child_idx != -1:
                    # 計算中心畫紅點並排除重複的框
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        center = np.array([cx, cy])
                        isPass = False
                        for old_center in cards_center:
                            if np.linalg.norm(center - old_center) < 50:
                                isPass = True
                        if isPass:
                            continue

                        cards_center.append(center)

                        inner_w = int(w * 1.1) # 向外擴 10%
                        inner_h = int(h * 1.1)
                        inner_x = x + int((w - inner_w) / 2)
                        inner_y = y + int((h - inner_h) / 2)
                        
                        detected_cards.append({
                            "center": center,
                            "approx": approx,
                            "box": (inner_x, inner_y, inner_w, inner_h)
                        })

        if len(detected_cards) > 0:
            # 1. 依據 y 座標進行初步分排 (從上到下)
            detected_cards.sort(key=lambda c: c["center"][1])
            
            rows = []
            current_row = [detected_cards[0]]
            for card in detected_cards[1:]:
                # 如果 y 與該排第一張差異小於 150，視為同一排
                if card["center"][1] - current_row[0]["center"][1] < 150:
                    current_row.append(card)
                else:
                    rows.append(current_row)
                    current_row = [card]
            rows.append(current_row)
            
            # 2. 從下到上處理各排 (反轉 rows)
            rows = rows[::-1]
            
            final_cards = []
            w_f = display_frame.shape[1]
            
            for row in rows:
                N = len(row)
                if N <= 5:
                    # 每排只有 4 張以下時，排除這排的所有框
                    continue
                
                # 該排由右到左排序 (x 遞減)
                row.sort(key=lambda c: c["center"][0], reverse=True)
                
                # 計算該排的平均寬高與Y座標
                avg_w = np.mean([c["box"][2] for c in row])
                avg_h = np.mean([c["box"][3] for c in row])
                avg_y = np.mean([c["center"][1] for c in row])
                
                x_arr = [c["center"][0] for c in row]
                dx_arr = [x_arr[i] - x_arr[i+1] for i in range(N-1)]
                space_x = np.median(dx_arr) if len(dx_arr) > 0 else 150
                
                # 保護機制：避免重疊導致的錯誤間距
                if space_x < avg_w * 0.5:
                    space_x = avg_w * 1.1 
                    
                best_error = float('inf')
                best_X_rightmost = x_arr[0]
                
                # 尋找 7 個插槽的最佳對齊位置
                for i in range(N):
                    for slot_idx in range(7):
                        cand_X_rightmost = x_arr[i] + slot_idx * space_x
                        slots_x = [cand_X_rightmost - k * space_x for k in range(7)]
                        
                        # 誤差計算: 偵測到的牌到最近插槽的距離總和
                        error = sum(min(abs(x - sx) for sx in slots_x) for x in x_arr)
                        # 懲罰項: 讓這 7 張牌整體盡量置中
                        center_offset = abs(np.mean(slots_x) - w_f / 2)
                        error += center_offset * 0.01
                        
                        if error < best_error:
                            best_error = error
                            best_X_rightmost = cand_X_rightmost
                            
                ideal_slots_x = [best_X_rightmost - k * space_x for k in range(7)]
                
                for slot_x in ideal_slots_x:
                    closest_card = min(row, key=lambda c: abs(c["center"][0] - slot_x))
                    if abs(closest_card["center"][0] - slot_x) < space_x * 0.4:
                        if "is_guessed" not in closest_card:
                            closest_card["is_guessed"] = False
                        final_cards.append(closest_card)
                    else:
                        # 自動猜測其餘的位置
                        cx, cy = int(slot_x), int(avg_y)
                        w, h = int(avg_w), int(avg_h)
                        bx, by = int(cx - w/2), int(cy - h/2)
                        
                        approx = np.array([
                            [[bx, by]],
                            [[bx, by + h]],
                            [[bx + w, by + h]],
                            [[bx + w, by]]
                        ], dtype=np.int32)
                        
                        final_cards.append({
                            "center": np.array([cx, cy]),
                            "approx": approx,
                            "box": (bx, by, w, h),
                            "is_guessed": True
                        })
                        
            detected_cards = final_cards

        # 先畫沒有數字的框與紅點
        for card in detected_cards:
            approx = card["approx"]
            center = card["center"]
            c_tuple = (int(center[0]), int(center[1]))
            
            # 畫框 (自動猜測的框用淺藍色顯示)
            color = (255, 255, 0) if card.get("is_guessed", False) else (0, 255, 0)
            cv2.drawContours(display_frame, [approx], -1, color, 3)
            # 畫中心紅點
            cv2.circle(display_frame, c_tuple, 7, (0, 0, 255), -1)

        # 顯示尚未編號的畫面
        h_f, w_f = display_frame.shape[:2]
        if w_f > 1280:
            show_frame = cv2.resize(display_frame, (1280, int(h_f * (1280/w_f))))
        else:
            show_frame = display_frame
        cv2.imshow('Tarot Tracker - Centroid Dot', show_frame)
        cv2.waitKey(10) # 讓畫面刷新

        if len(detected_cards) > 0:
            # 讓使用者輸入
            start_idx_str = input(f"第 {frame_count} 幀：請輸入最右下角初始值 (直接 Enter 跳過，輸入 q 結束): ")
            if start_idx_str.lower() == 'q':
                break
            if start_idx_str.strip() == '':
                continue
                
            try:
                start_idx = int(start_idx_str.strip())
            except ValueError:
                print("輸入無效，跳過此幀。")
                continue
                
            # 有效輸入後，畫數字並存檔
            for i, card in enumerate(detected_cards):
                num = start_idx + i
                center = card["center"]
                c_tuple = (int(center[0]), int(center[1]))
                
                # 畫數字編號
                text = str(num)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                text_x = c_tuple[0] - text_size[0] // 2
                text_y = c_tuple[1] + text_size[1] // 2
                
                # 畫黑色背景框讓字體更明顯
                cv2.rectangle(display_frame, 
                             (text_x - 5, text_y - text_size[1] - 5), 
                             (text_x + text_size[0] + 5, text_y + 5), 
                             (0, 0, 0), -1)
                # 畫白色字體
                cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                # 裁切並儲存原始圖片 (從沒有畫線的 frame 裁切)
                bx, by, bw, bh = card["box"]
                bx_start = max(0, bx)
                by_start = max(0, by)
                bx_end = min(frame.shape[1], bx + bw)
                by_end = min(frame.shape[0], by + bh)
                
                if bx_end > bx_start and by_end > by_start:
                    cropped = frame[by_start:by_end, bx_start:bx_end]
                    # 儲存到 OUTPUT_DIR
                    filename = os.path.join(OUTPUT_DIR, f"{num}_{frame_count}.png")
                    cv2.imwrite(filename, cropped)

            # 更新顯示有數字的畫面
            if w_f > 1280:
                show_frame = cv2.resize(display_frame, (1280, int(h_f * (1280/w_f))))
            else:
                show_frame = display_frame
            cv2.imshow('Tarot Tracker - Centroid Dot', show_frame)
            cv2.waitKey(10)
        else:
            # 沒偵測到牌，直接偵測按鍵
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("處理結束。")


if __name__ == "__main__":
    process_video_visual(VIDEO_PATH)