import os
import json
import glob
from PIL import Image
from tqdm import tqdm

def yolo_to_coco(image_dir, label_dir, output_json, class_names):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories
    for i, name in enumerate(class_names):
        coco_data["categories"].append({
            "id": i,
            "name": name,
            "supercategory": "tarot"
        })
        
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_files.sort()
    
    ann_id = 0
    for img_id, img_path in enumerate(tqdm(image_files, desc="Converting")):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        w, h = img.size
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_w = float(parts[3]) * w
            box_h = float(parts[4]) * h
            
            # COCO format: [x_min, y_min, width, height]
            x_min = x_center - box_w / 2
            y_min = y_center - box_h / 2
            
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x_min, y_min, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            ann_id += 1
            
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)
    print(f"Successfully saved COCO annotations to {output_json}")

def yolo_to_coco_list(image_files, label_dir, output_json, class_names):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for i, name in enumerate(class_names):
        coco_data["categories"].append({"id": i, "name": name, "supercategory": "tarot"})
        
    ann_id = 0
    for img_id, img_path in enumerate(tqdm(image_files, desc=f"Converting {os.path.basename(output_json)}")):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        w, h = img.size
        
        coco_data["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})
        
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            class_id, x_c, y_c, bw, bh = map(float, parts)
            x_min = (x_c - bw/2) * w
            y_min = (y_c - bh/2) * h
            coco_data["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": int(class_id),
                "bbox": [x_min, y_min, bw*w, bh*h], "area": bw*w*bh*h, "iscrowd": 0
            })
            ann_id += 1
            
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=4)

if __name__ == "__main__":
    # Setup paths
    image_dir = "C:/Users/User/Desktop/TarotTrainingData/dataset_train"
    label_dir = "C:/Users/User/Desktop/TarotTrainingData/dataset_train" # They are in the same dir
    
    # Organize for RT-DETRv2 (create a dedicated coco folder)
    coco_root = "C:/Users/User/Desktop/TarotTrainingData/coco_tarot"
    os.makedirs(os.path.join(coco_root, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(coco_root, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(coco_root, "annotations"), exist_ok=True)
    
    # Get class names from TarotImg (alphabetical order as in Generator)
    tarot_img_dir = "./Data/TarotImg"
    tarot_images = glob.glob(os.path.join(tarot_img_dir, "*.webp"))
    tarot_images.sort()
    class_names = [str(i) for i in range(78*2+1)]
    
    # Get all image files
    all_image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    all_image_files.sort()
    
    # Split: 90% train, 10% val
    import random
    random.seed(42)
    random.shuffle(all_image_files)
    split_idx = int(len(all_image_files) * 0.9)
    train_files = all_image_files[:split_idx]
    val_files = all_image_files[split_idx:]
    
    # 1. Convert Train
    output_train_json = os.path.join(coco_root, "annotations/instances_train.json")
    yolo_to_coco_list(train_files, label_dir, output_train_json, class_names)
    
    # 2. Convert Val
    output_val_json = os.path.join(coco_root, "annotations/instances_val.json")
    yolo_to_coco_list(val_files, label_dir, output_val_json, class_names)
    
    # 3. Copy images
    import shutil
    print("Organizing train images...")
    for img_p in train_files:
        shutil.copy(img_p, os.path.join(coco_root, "images/train", os.path.basename(img_p)))
        
    print("Organizing val images...")
    for img_p in val_files:
        shutil.copy(img_p, os.path.join(coco_root, "images/val", os.path.basename(img_p)))
        
    print("Done! Dataset organized in ./coco_tarot")
