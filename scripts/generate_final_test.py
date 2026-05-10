import numpy as np
from PIL import Image, ImageDraw
import random
from pathlib import Path

def create_plate(draw, cx, cy, r, label, colony_range=(20, 40)):
    # Agar color
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(45, 42, 38), outline=(90, 90, 95), width=5)
    
    # Colonies (place some at edges to test expansion/margin)
    num = random.randint(*colony_range)
    for i in range(num):
        angle = random.uniform(0, 2 * np.pi)
        if i < 10: # Edge colonies: 2-5 pixels from agar edge
            dist = r - random.uniform(2, 6)
        else:
            dist = np.sqrt(random.uniform(0, 1)) * (r - 15)
        col_x, col_y = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
        cr = random.randint(5, 10)
        # Creamy colony color
        color = (random.randint(230, 255), random.randint(220, 245), random.randint(190, 215))
        draw.ellipse([col_x-cr, col_y-cr, col_x+cr, col_y+cr], fill=color)

def generate_final_touching_test(output_path="sandbox/final_touching_test.jpg"):
    rows, cols = 2, 3
    plate_radius = 500
    margin = 50
    
    # Image size to fit exactly touching plates plus a small border
    w = (cols * plate_radius * 2) + (margin * 2)
    h = (rows * plate_radius * 2) + (margin * 2)
    
    img = Image.new("RGB", (w, h), (20, 20, 25))
    draw = ImageDraw.Draw(img)
    
    for r in range(rows):
        for c in range(cols):
            # Calculate centers so they are touching but not overlapping
            cx = margin + (c * plate_radius * 2) + plate_radius
            cy = margin + (r * plate_radius * 2) + plate_radius
            
            # Add tiny 2px gap to simulate real tight placement
            create_plate(draw, cx, cy, plate_radius - 1, f"P_{r}{c}")
            
    img.save(output_path)
    print(f"Final touching test image saved to {output_path} ({w}x{h})")

if __name__ == "__main__":
    Path("sandbox").mkdir(exist_ok=True)
    generate_final_touching_test()
