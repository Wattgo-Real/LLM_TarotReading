import os
import random
import glob
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance

import numpy as np

class TarotDataGenerator:
    def __init__(self, img_dir="./TarotImg", canvas_size=(800, 600)):
        self.img_dir = img_dir
        self.canvas_size = canvas_size
        #self.image_paths = glob.glob(os.path.join(img_dir, "*.webp"))
        #self.image_paths.sort()
        #self.class_map = {os.path.basename(path): i for i, path in enumerate(self.image_paths)}
        #self.class_map[78*2] = 78*2
        # self.id_to_name = {i: os.path.basename(path) for i, path in enumerate(self.image_paths)}

        self.image_paths_realdata = glob.glob(os.path.join(img_dir, "*.png"))
        self.image_paths = [[it for it in self.image_paths_realdata if os.path.basename(it).split("_")[2] == str(i)] for i in range(79)]
        
        print(f"Loaded {len(self.image_paths)} tarot images.")

    def _add_shadows(self, image):
        """Adds random shadows, including a rectangular shadow mimicking a phone."""
        shadow_layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow_layer)
        
        # 1. Environmental Bottom Shadow
        if random.random() > 0.2:
            shadow_height = random.randint(100, 300)
            max_alpha = random.randint(50, 150)
            for y in range(self.canvas_size[1] - shadow_height, self.canvas_size[1]):
                alpha = int(max_alpha * ((y - (self.canvas_size[1] - shadow_height)) / shadow_height))
                draw.line([(0, y), (self.canvas_size[0], y)], fill=(0, 0, 0, alpha))
        
        # 2. Rectangular "Phone" Shadow
        # Simulates the shadow of the phone/camera held above
        if random.random() > 0.1: # 90% chance of phone shadow
            # Phone shadow is usually at the bottom or coming from a corner
            pw = random.randint(200, 400)
            ph = random.randint(300, 500)
            # Positioned at bottom center or slightly offset
            px = random.randint(self.canvas_size[0]//4, self.canvas_size[0]*3//4 - pw)
            py = self.canvas_size[1] - random.randint(ph//2, ph)
            
            # Draw phone body shadow
            draw.rectangle([px, py, px+pw, py+ph], fill=(0, 0, 0, random.randint(60, 120)))
            
        # 3. Random blob shadow (extra noise)
        if random.random() > 0.5:
            x = random.randint(0, self.canvas_size[0])
            y = random.randint(0, self.canvas_size[1])
            r = random.randint(100, 300)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 0, 0, random.randint(30, 80)))
            
        # Blur the shadow layer heavily
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=random.randint(30, 60)))
        return Image.alpha_composite(image.convert("RGBA"), shadow_layer).convert("RGB")

    def _apply_perspective(self, card, tilt_x=0.1, tilt_type='top'):
        """
        Applies a trapezoidal perspective transform to a card.
        tilt_x: how much the width changes (0 to 0.2 recommended)
        tilt_type: 'top' (top side narrower) or 'bottom' (bottom side narrower)
        """
        w, h = card.size
        offset = int(w * tilt_x)
        
        if tilt_type == 'top':
            # Target quad: Top side is narrower
            # (top_left, top_right, bottom_right, bottom_left)
            quad = (offset, 0, w - offset, 0, w, h, 0, h)
        else:
            # Target quad: Bottom side is narrower
            quad = (0, 0, w, 0, w - offset, h, offset, h)
            
        # We use transform with Image.QUAD. 
        # The quad argument is the coordinates of the source image to be mapped to the destination.
        # But we want to map the source RECT to a destination QUAD. 
        # In Pillow, transform(QUAD) maps a QUAD in the SOURCE to a RECT in the DEST.
        # So we do the inverse or just expand the canvas.
        
        # For simplicity, we can use card.transform with a perspective matrix or quad
        # Actually, let's use a simpler method: mapping source corners to target corners.
        # Pillow's QUAD transform: data = (x0, y0, x1, y1, x2, y2, x3, y3) 
        # representing the quad in the SOURCE image that is mapped to the output rectangle.
        
        # To get a trapezoid where the output is narrower at the top, 
        # we define a quad in the source that is WIDER at the top? No, it's confusing.
        
        # Better: create a larger canvas, paste card, transform the whole thing, then crop.
        # Or use mesh transform. Let's use the standard QUAD:
        # quad maps (0,0, 0,h, w,h, w,0) in the source to the 4 corners of the output.
        # Wait, the documentation says: data is (x0, y0, ..., x3, y3) quad in source.
        
        # Let's use a perspective matrix for more control.
        # But QUAD is easier if we just want a trapezoid.
        # If source quad is (0,0, w,0, w,h, 0,h) and output is rect, it's no-op.
        # If source quad is (-offset, 0, w+offset, 0, w, h, 0, h), the output rect will look like a trapezoid.
        
        card_with_padding = Image.new("RGBA", (w + 2*offset, h), (0,0,0,0))
        card_with_padding.paste(card, (offset, 0))
        wp, hp = card_with_padding.size
        
        if tilt_type == 'top':
            # Source quad that will be "stretched" to fill the output rectangle
            # This makes the top look narrower in the result.
            source_quad = (0, 0, offset, hp, wp - offset, hp, wp, 0)
        else:
            source_quad = (offset, 0, 0, hp, wp, hp, wp - offset, 0)
            
        transformed = card_with_padding.transform((wp, hp), Image.QUAD, source_quad, Image.BICUBIC)
        return transformed

    def _add_random_glare(self, img, max_glare_size=0.5):
        """
        img: PIL Image
        max_glare_size: 反光最大佔圖比例
        """
        img_np = np.array(img).astype(np.float32)
        h, w, _ = img_np.shape

        # 1️⃣ 隨機中心位置
        cx = random.randint(0, w)
        cy = random.randint(0, h)

        # 2️⃣ 隨機反光大小
        radius = int(min(h, w) * random.uniform(0.1, max_glare_size))

        # 3️⃣ 建立座標 grid
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # 4️⃣ 建立圓形 mask（越中心越亮）
        mask = np.clip(1 - dist / radius, 0, 1)

        # 5️⃣ 隨機顏色（偏白或帶色）
        color = np.array([
            random.uniform(0, 1.0),  # R
            random.uniform(0, 1.0),  # G
            random.uniform(0, 1.0)   # B
        ])

        # 6️⃣ 強度
        intensity = random.uniform(50, 120)

        # 7️⃣ 套用 glare
        for c in range(3):
            img_np[..., c] += mask * color[c] * intensity

        # 8️⃣ clip
        img_np = np.clip(img_np, 0, 255)

        # 9️⃣ 轉回 PIL 並加模糊（更自然）
        result = Image.fromarray(img_np.astype(np.uint8))

        return result

    def generate_sample(self):
        # Base background
        if random.random() > 0.5:
            bg = Image.new("RGB", self.canvas_size, (random.randint(40, 70), random.randint(40, 70), random.randint(40, 70)))
        else:
            bg = Image.new("RGB", self.canvas_size, (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220)))
        # 1. Determine Global parameters for the whole image
        # Consistent scale for all cards in this image
        global_scale = random.uniform(0.7, 0.85) 
        
        # Consistent perspective tilt
        tilt_amount = random.uniform(0.00, 0.15)
        tilt_direction = random.choice(['top'])
        
        scale = 0.6 + 0.4 * random.random()
        center_w = random.randint(-100 + self.canvas_size[0]//2, 100 + self.canvas_size[0]//2)
        region_width = int(self.canvas_size[0] / 4 * scale)
        regions = [(center_w + int(region_width * -3 * scale), center_w + int(region_width * -1 * scale)), 
                   (center_w + int(region_width * -1  * scale), center_w + int(region_width * 1 * scale)), 
                   (center_w + int(region_width * 1 * scale), center_w + int(region_width * 3 * scale))]
        
        active_regions = [r for r in regions if random.random() > 0.3]
        if not active_regions: active_regions = [random.choice(regions)]
        
        selected_paths = random.sample([i for i in range(79)], len(active_regions))
        labels = []
        
        # Layer for cards to allow alpha compositing before shadows
        card_layer = Image.new("RGBA", self.canvas_size, (0, 0, 0, 0))
        
        for i, region in enumerate(active_regions):
            path = random.sample(self.image_paths[selected_paths[i]], 1)[0]
            class_id = selected_paths[i]
            card = Image.open(path).convert("RGBA").crop((30,0,370,600))
            factor = random.random()/2 + 0.6
            enhancer = ImageEnhance.Brightness(card)
            card = enhancer.enhance(factor)
            #card = self._add_random_glare(card, max_glare_size=2)
            
            if class_id == 78:
                class_id = 78 * 2
            # Apply reversal
            if random.randint(0,1) == 1 and class_id != 78 * 2:
                card = card.rotate(180)
                class_id += 78

            # Apply uniform scaling
            new_w = int(region_width * global_scale)
            aspect_ratio = card.height / card.width
            new_h = int(new_w * aspect_ratio)
            card = card.resize((new_w, new_h), Image.LANCZOS)
            
            # Apply uniform perspective (trapezoid)
            card = self._apply_perspective(card, tilt_x=tilt_amount, tilt_type=tilt_direction)
            
            # Random subtle rotation
            angle = random.randint(-10, 10)
            card = card.rotate(angle, expand=True, resample=Image.BICUBIC)
            
            cw, ch = card.size
            region_center_x = (region[0] + region[1]) // 2
            x_pos = region_center_x - (cw // 2) + random.randint(-15, 15)
            y_pos = (self.canvas_size[1] // 2) - (ch // 2) + random.randint(-40, 40)
            
            # Clamping
            x_pos = max(0, min(x_pos, self.canvas_size[0] - cw))
            y_pos = max(0, min(y_pos, self.canvas_size[1] - ch))
            
            card_layer.paste(card, (x_pos, y_pos), card)
            
            # Calculate BBox (YOLO)
            # Use bbox to get non-transparent area
            bbox = card.getbbox()
            if bbox:
                # Real extent on canvas
                real_x1 = x_pos + bbox[0]
                real_y1 = y_pos + bbox[1]
                real_x2 = x_pos + bbox[2]
                real_y2 = y_pos + bbox[3]
                
                w_box = real_x2 - real_x1
                h_box = real_y2 - real_y1
                x_center = (real_x1 + w_box/2) / self.canvas_size[0]
                y_center = (real_y1 + h_box/2) / self.canvas_size[1]
                norm_w = w_box / self.canvas_size[0]
                norm_h = h_box / self.canvas_size[1]
                
                labels.append([class_id, x_center, y_center, norm_w, norm_h])

        # Composite cards onto bg
        combined = Image.alpha_composite(bg.convert("RGBA"), card_layer).convert("RGB")
        
        # Add environmental shadows
        final_img = self._add_shadows(combined)
        
        return final_img, labels

    def save_sample(self, output_dir, filename):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img, labels = self.generate_sample()
        img_path = os.path.join(output_dir, f"{filename}.jpg")
        img.save(img_path)
        label_path = os.path.join(output_dir, f"{filename}.txt")
        with open(label_path, "w") as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
        return img_path, label_path

if __name__ == "__main__":
    generator = TarotDataGenerator(img_dir="./TarotImg")
    output_dir = "C:/Users/User/Desktop/dataset_train"
    data_num = 10000
    print(f"Generating {data_num} enhanced samples in {output_dir}...")
    for i in range(data_num):
        img_p, _ = generator.save_sample(output_dir, f"enhanced_{i}")
        print(f"Saved: {img_p}")
