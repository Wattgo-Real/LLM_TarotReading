import os
import glob
import numpy as np
from PIL import Image

def get_crop_margins(img):
    """
    Detect the margins of the white borders by probing inwards from the center of the 4 edges
    and looking for the scanner_background -> white_border -> card_black_outline transition.
    """
    gray = img.convert('L')
    arr = np.array(gray)
    h, w = arr.shape
    
    center_y = h // 2
    center_x = w // 2
    
    # 1. Probe top margin
    top = 0
    for y in range(h // 10):
        val = np.mean(arr[y, center_x - 2 : center_x + 3])
        if val > 230:
            seen_white = True
            top = y + 1
        else:
            break
            
    # 2. Probe bottom margin
    bottom = h
    for y in range(h - 1, h - h // 10, -1):
        val = np.mean(arr[y, center_x - 2 : center_x + 3])
        if val > 230:
            bottom = y
        else:
            break

    # 3. Probe left margin
    left = 0
    for x in range(w // 10):
        val = np.mean(arr[center_y - 2 : center_y + 3, x])
        if val > 230:
            left = x + 1
        else:
            break

    # 4. Probe right margin
    right = w
    for x in range(w - 1, w - w // 10, -1):
        val = np.mean(arr[center_y - 2 : center_y + 3, x])
        if val > 230:
            right = x
        else:
            break

    return top, bottom, left, right

def main():
    img_dir = "Data/TarotScanAll"
    image_paths = glob.glob(os.path.join(img_dir, "*.png"))
    image_paths.sort()

    #image_paths = ["./Data/TarotScanAll/29.png"]
    
    print(f"Found {len(image_paths)} images to process.")
    
    cropped_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                top, bottom, left, right = get_crop_margins(img)
                
                # Check if cropping is actually needed
                if top > 0 or bottom < h or left > 0 or right < w:
                    # Perform crop
                    cropped_img = img.crop((left+1, top+1, right-1, bottom-1))
                    # Save back to the original file
                    cropped_img.save(path)
                    print(f"Cropped {os.path.basename(path)}: Original ({w}x{h}) -> Cropped ({right-left}x{bottom-top}) | Left margin: {left}px, Top margin: {top}px, Right margin: {w-right}px, Bottom margin: {h-bottom}px")
                    cropped_count += 1
                else:
                    print(f"Skipped {os.path.basename(path)}: No white borders detected.")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"\nCompleted! Total cropped images: {cropped_count}/{len(image_paths)}")

if __name__ == "__main__":
    main()
