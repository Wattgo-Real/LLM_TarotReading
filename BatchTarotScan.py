import cv2
import numpy as np
import os
import glob

def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    result, nparray = cv2.imencode(ext, img)
    if result:
        with open(path, mode='wb') as f:
            nparray.tofile(f)

# Parameters from MP4ToTarotImg.py - Adjusted for potential high-res images
CANNY_LOW = 0
CANNY_HIGH = 100
# Removing strict area constraints initially to see what's detected
MIN_AREA = 5000 
MAX_AREA = 5000000 
TARGET_ASPECT_RATIO = 1.6
ASPECT_RATIO_TOLERANCE = 0.5 # More tolerant

INPUT_DIR = 'TarotScan'
OUTPUT_DIR = 'TarotScanAll'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all 'r' files
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*r.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*r.png")))

total_card_count = 0

for file_path in files:
    print(f"Processing {file_path}...")
    frame = imread_unicode(file_path)
    if frame is None:
        print(f"Failed to load {file_path}")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_cards = []
    cards_center = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Must be a 4-sided convex polygon
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
                
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = max(h, w) / float(min(h, w)) # Handle both orientations

            if not (MIN_AREA < area < MAX_AREA):
                continue
                
            if not (1.0 < aspect_ratio < 2.5): # Very broad aspect ratio
                continue

            # In MP4ToTarotImg, it looked for nested contours.
            # Let's see if we can find cards without that strict check first,
            # or at least print how many nested ones we find.
            child_idx = hierarchy[i][2]
            parent_idx = hierarchy[i][3]

            # If we want the "scanning method" of MP4ToTarotImg, it used hierarchy.
            # But maybe the images don't have the same hierarchy.
            # Let's relax this but keep the logic if possible.
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center = np.array([cx, cy])
                
                is_duplicate = False
                for old_center in cards_center:
                    if np.linalg.norm(center - old_center) < 100: # Larger distance for duplicates
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

                cards_center.append(center)
                
                # Expand box slightly like in the original script
                inner_w = int(w * 1.05)
                inner_h = int(h * 1.05)
                inner_x = x - int((inner_w - w) / 2)
                inner_y = y - int((inner_h - h) / 2)
                
                detected_cards.append({
                    "center": center,
                    "approx": approx,
                    "box": (inner_x, inner_y, inner_w, inner_h),
                    "area": area
                })

    print(f"  Detected {len(detected_cards)} potential cards.")
    if not detected_cards:
        continue

    # Sorting logic
    # Group by rows. The 150 threshold might need adjustment.
    # Let's dynamically determine the threshold based on average card height.
    avg_card_h = np.mean([c["box"][3] for c in detected_cards])
    row_threshold = avg_card_h * 0.5
    
    detected_cards.sort(key=lambda c: c["center"][1])
    
    rows = []
    current_row = [detected_cards[0]]
    for card in detected_cards[1:]:
        if card["center"][1] - current_row[0]["center"][1] < row_threshold:
            current_row.append(card)
        else:
            rows.append(current_row)
            current_row = [card]
    rows.append(current_row)
    
    # Reverse rows (bottom to top)
    rows = rows[::-1]
    print(f"  Organized into {len(rows)} rows.")
    
    final_cards = []
    img_h, img_w = frame.shape[:2]
    
    for r_idx, row in enumerate(rows):
        # Sort row by x reverse (right to left)
        row.sort(key=lambda c: c["center"][0], reverse=True)
        
        N = len(row)
        print(f"    Row {r_idx}: {N} cards.")
        
        # We want exactly 5 cards per row per user request.
        # If there are fewer than 5, we fill. If more, we might have noise?
        # Let's use the slot alignment logic for 5 slots.
        
        avg_w = np.mean([c["box"][2] for c in row])
        avg_h = np.mean([c["box"][3] for c in row])
        avg_y = np.mean([c["center"][1] for c in row])
        
        if N >= 2:
            x_arr = [c["center"][0] for c in row]
            dx_arr = [x_arr[i] - x_arr[i+1] for i in range(N-1)]
            space_x = np.median(dx_arr)
        else:
            space_x = avg_w * 1.2
        
        if space_x < avg_w * 0.8:
            space_x = avg_w * 1.2
            
        best_error = float('inf')
        best_X_rightmost = x_arr[0] if N > 0 else img_w * 0.8
        
        if N > 0:
            for i in range(N):
                for slot_idx in range(5):
                    cand_X_rightmost = x_arr[i] + slot_idx * space_x
                    slots_x = [cand_X_rightmost - k * space_x for k in range(5)]
                    error = sum(min(abs(x - sx) for sx in slots_x) for x in x_arr)
                    center_offset = abs(np.mean(slots_x) - img_w / 2)
                    error += center_offset * 0.01
                    if error < best_error:
                        best_error = error
                        best_X_rightmost = cand_X_rightmost
        
        ideal_slots_x = [best_X_rightmost - k * space_x for k in range(5)]
        
        for slot_x in ideal_slots_x:
            if N > 0:
                closest_card = min(row, key=lambda c: abs(c["center"][0] - slot_x))
                if abs(closest_card["center"][0] - slot_x) < space_x * 0.4:
                    final_cards.append(closest_card)
                    continue
            
            # Guessing
            cx, cy = int(slot_x), int(avg_y)
            w, h = int(avg_w), int(avg_h)
            bx, by = int(cx - w/2), int(cy - h/2)
            final_cards.append({
                "center": np.array([cx, cy]),
                "box": (bx, by, w, h),
                "is_guessed": True
            })

    # Label and save
    for card in final_cards:
        bx, by, bw, bh = card["box"]
        bx_start = max(0, bx)
        by_start = max(0, by)
        bx_end = min(frame.shape[1], bx + bw)
        by_end = min(frame.shape[0], by + bh)
        
        if bx_end > bx_start and by_end > by_start:
            cropped = frame[by_start:by_end, bx_start:bx_end]
            out_name = os.path.join(OUTPUT_DIR, f"{total_card_count}.png")
            imwrite_unicode(out_name, cropped)
            total_card_count += 1


print(f"Finished! Processed {total_card_count} cards into {OUTPUT_DIR}.")
