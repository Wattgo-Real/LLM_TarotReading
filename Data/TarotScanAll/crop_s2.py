import os
import glob
from PIL import Image

def main():
    target_dir = r"C:\Users\User\Desktop\TarotTrainingData\TarotScanS2"
    
    # Supported image extensions
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(target_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(target_dir, ext.upper())))
    
    image_paths = list(set(image_paths)) # Remove duplicates
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {target_dir}")
        return
        
    print(f"Found {len(image_paths)} images in {target_dir} to process.")
    
    processed_count = 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                if w > 10 and h > 10:
                    # Crop 5 pixels off all edges: left, top, right, bottom
                    cropped_img = img.crop((35, 5, w - 35, h - 5))
                    cropped_img.save(path)
                    print(f"Cropped {os.path.basename(path)}: Original ({w}x{h}) -> Cropped ({w-10}x{h-10})")
                    processed_count += 1
                else:
                    print(f"Skipping {os.path.basename(path)}: Image size too small ({w}x{h})")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    print(f"\nCompleted! Successfully processed {processed_count}/{len(image_paths)} images.")

if __name__ == "__main__":
    main()
