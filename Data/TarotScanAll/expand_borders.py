import os
import glob
import numpy as np
from PIL import Image

# ==================== CONFIGURATION ====================
# Specify how many pixels to expand on each side:
EXPAND_TOP = 25
EXPAND_BOTTOM = 100
EXPAND_LEFT = 25
EXPAND_RIGHT = 25

# Color Correction Options:
REMOVE_BLUE_TINT = True  # If True, reduces the blue channel if it exceeds R or G to remove blue color cast
MANUAL_COLOR = (0, 7, 36)  # e.g., (245, 245, 240) to manually force a color, or None to auto-detect

# Bar Image Options:
BAR_PATH = "Data/TarotScanAll/TarotScan/bar.png"
BAR_WIDTH = 280          # Set to positive integer to resize, or -1 to use original width
BAR_HEIGHT = -1          # Set to positive integer to resize, or -1 to keep aspect ratio
# =======================================================

def expand_image_borders(img, top=EXPAND_TOP, bottom=EXPAND_BOTTOM, left=EXPAND_LEFT, right=EXPAND_RIGHT):
    """
    Expands the image borders outward by filling them with the detected edge color of the image.
    Applies color correction if enabled.
    """
    # Ensure image is in RGB mode to work with colors
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    w, h = img.size
    
    if MANUAL_COLOR is not None:
        fill_color = MANUAL_COLOR
    else:
        arr = np.array(img)
        # Gather pixels along the very edge to determine the solid background color
        edge_pixels = np.concatenate([
            arr[0, :, :],       # Top edge row
            arr[-1, :, :],      # Bottom edge row
            arr[:, 0, :],       # Left edge col
            arr[:, -1, :]       # Right edge col
        ])
        
        # Use median color to resist noise
        median_color = np.median(edge_pixels, axis=0).astype(int)
        r, g, b = median_color
        
        # Color correction: neutralize blue cast if blue is higher than red or green
        if REMOVE_BLUE_TINT and (b > r or b > g):
            b = int((r + g) / 2)
            
        fill_color = (int(r), int(g), int(b))
    
    # Calculate new dimensions
    new_w = w + left + right
    new_h = h + top + bottom
    
    # Create the new canvas filled with the border color
    expanded_img = Image.new("RGB", (new_w, new_h), fill_color)
    
    # Paste original card image
    expanded_img.paste(img, (left, top))
    
    # Add the bar.png image if it exists
    if os.path.exists(BAR_PATH):
        try:
            bar_img = Image.open(BAR_PATH)
            # Ensure in RGBA mode for transparency support
            if bar_img.mode != 'RGBA':
                bar_img = bar_img.convert('RGBA')
                
            bar_w, bar_h = bar_img.size
            
            # Resize bar image based on settings
            if BAR_WIDTH > 0 and BAR_HEIGHT > 0:
                bar_img = bar_img.resize((BAR_WIDTH, BAR_HEIGHT), Image.Resampling.LANCZOS)
            elif BAR_WIDTH > 0:
                ratio = BAR_WIDTH / bar_w
                bar_img = bar_img.resize((BAR_WIDTH, int(bar_h * ratio)), Image.Resampling.LANCZOS)
            elif BAR_HEIGHT > 0:
                ratio = BAR_HEIGHT / bar_h
                bar_img = bar_img.resize((int(bar_w * ratio), BAR_HEIGHT), Image.Resampling.LANCZOS)
            
            bar_w, bar_h = bar_img.size
            
            # Position at horizontal center, 30 pixels above bottom edge
            bar_x = (new_w - bar_w) // 2
            bar_y = new_h - bar_h - 30
            
            # Paste using itself as the mask to preserve transparency
            expanded_img.paste(bar_img, (bar_x, bar_y), bar_img)
        except Exception as e:
            print(f"Warning: Could not process bar image at {BAR_PATH}: {e}")
    else:
        print(f"Warning: Bar image not found at {BAR_PATH}")
    
    # Apply dynamic blue channel reduction: the darker the overall image, the more the blue channel is reduced (max 30)
    arr_exp = np.array(expanded_img).astype(np.float32)
    avg_brightness = np.mean(arr_exp)  # Overall brightness of the image
    reduction = 30.0 * (1.0 - avg_brightness / 255.0)
    
    # Subtract reduction from the Blue channel (index 2)
    arr_exp[:, :, 2] = np.clip(arr_exp[:, :, 2] - reduction, 0, 255)
    expanded_img = Image.fromarray(arr_exp.astype(np.uint8))
    
    print(f"  Overall Brightness: {avg_brightness:.1f} | Blue Channel Reduction: -{reduction:.1f}")

    return expanded_img, fill_color

def main():
    img_dir = "Data/TarotScanAll"
    image_paths = glob.glob(os.path.join(img_dir, "*.png"))
    image_paths.sort()
    
    # Allow processing only 0.png if specified, or all of them
    # To process all files, comment out the line below:

    print(f"Found {len(image_paths)} images to process.")
    print("Expansion settings:")
    print(f"  Top:    {EXPAND_TOP} px")
    print(f"  Bottom: {EXPAND_BOTTOM} px")
    print(f"  Left:   {EXPAND_LEFT} px")
    print(f"  Right:  {EXPAND_RIGHT} px")
    print(f"Color Correction (REMOVE_BLUE_TINT): {REMOVE_BLUE_TINT}")
    
    processed_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                expanded_img, fill_color = expand_image_borders(img)
                
                # Save the expanded image back
                expanded_img.save(path)
                
                new_w, new_h = expanded_img.size
                print(f"Processed {os.path.basename(path)}: ({w}x{h}) -> ({new_w}x{new_h}) | Fill Color RGB: {fill_color}")
                processed_count += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"\nCompleted! Total processed images: {processed_count}/{len(image_paths)}")

if __name__ == "__main__":
    main()
