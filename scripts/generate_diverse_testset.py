import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path

def create_plate(draw, cx, cy, r, label, colony_range=(10, 35)):
    # Agar color
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(45, 42, 38), outline=(90, 90, 95), width=4)
    # Colonies (place some at edges)
    num = random.randint(*colony_range)
    for i in range(num):
        angle = random.uniform(0, 2 * np.pi)
        if i < 8: # Edge colonies
            dist = r - random.uniform(2, 8)
        else:
            dist = np.sqrt(random.uniform(0, 1)) * (r - 12)
        col_x, col_y = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
        cr = random.randint(4, 9)
        # Creamy colony color
        color = (random.randint(230, 255), random.randint(220, 245), random.randint(190, 215))
        draw.ellipse([col_x-cr, col_y-cr, col_x+cr, col_y+cr], fill=color)

def generate_diverse_testset(out_dir="sandbox/diverse_tests"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    scenarios = [
        # 1. Standard 2x3 Grid (Perfectly centered)
        ("case01_standard_2x3", 3000, 2000, 2, 3, 400, "center"),
        # 2. Touching Plates (Large radius)
        ("case02_touching", 3000, 2000, 2, 3, 500, "center"),
        # 3. Corner Aligned (Top-Left corner)
        ("case03_top_left_corner", 4000, 4000, 2, 3, 400, "top-left"),
        # 4. Mega Scan (Flatbed simulation 6000x2000)
        ("case04_mega_scan_6000x2000", 6000, 2000, 2, 3, 450, "center"),
        # 5. Large Plates filling frame
        ("case05_filling_frame", 2000, 1500, 2, 2, 350, "center"),
        # 6. Irregular Spacing (Simulated manual placement)
        ("case06_irregular", 3000, 2000, 2, 3, 350, "jitter"),
        # 7. Skewed/Rotated (8 degree rotation)
        ("case07_skewed", 3000, 2000, 2, 2, 400, "skewed"),
        # 8. Missing Plate (Slot A2 is empty)
        ("case08_missing_plate", 3000, 2000, 2, 3, 400, "missing_a2"),
        # 9. Portrait Orientation (3x2)
        ("case09_portrait_3x2", 2000, 4500, 3, 2, 400, "center"),
        # 10. Crowded 3x4 Grid
        ("case10_crowded_3x4", 4000, 3000, 3, 4, 350, "center"),
    ]

    for name, w, h, rows, cols, r, style in scenarios:
        img = Image.new("RGB", (w, h), (20, 20, 25))
        draw = ImageDraw.Draw(img)
        
        cell_w, cell_h = w // (cols if "corner" not in style else 4), h // (rows if "corner" not in style else 4)
        
        offset_x, offset_y = 0, 0
        if style == "top-left":
            offset_x, offset_y = 500, 500
        elif style == "bottom-right":
            offset_x, offset_y = w - (cols * 800), h - (rows * 800)
            cell_w, cell_h = 800, 800

        for row in range(rows):
            for col in range(cols):
                if style == "missing_a2" and row == 0 and col == 1:
                    continue
                
                if style == "jitter":
                    cx = col * 1000 + 500 + random.randint(-150, 150)
                    cy = row * 1000 + 500 + random.randint(-150, 150)
                elif style == "skewed":
                    angle = np.radians(8)
                    lx, ly = (col-0.5)*1000, (row-0.5)*1000
                    cx = int(lx * np.cos(angle) - ly * np.sin(angle) + w//2)
                    cy = int(lx * np.sin(angle) + ly * np.cos(angle) + h//2)
                else:
                    cx = col * cell_w + cell_w // 2 + offset_x
                    cy = row * cell_h + cell_h // 2 + offset_y
                
                create_plate(draw, cx, cy, r, f"{row}{col}")
        
        img.save(out_path / f"{name}.jpg")
    
    print(f"Generated 10 diverse test scenarios in {out_dir}")

if __name__ == "__main__":
    generate_diverse_testset()
