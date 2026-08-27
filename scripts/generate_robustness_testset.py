import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_plate(draw, cx, cy, r, label, colony_range=(10, 30)):
    # Draw agar
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r], fill=(45, 40, 35), outline=(100, 100, 105), width=4
    )
    # Draw colonies - intentionally placing some near the edge
    num = random.randint(*colony_range)
    for i in range(num):
        angle = random.uniform(0, 2 * np.pi)
        # Randomly place some exactly at the edge or very close
        # Edge colonies sit exactly 5 px inside the agar edge; the rest spread across the plate.
        dist = r - 5 if i < 5 else np.sqrt(random.uniform(0, 1)) * (r - 10)

        col_x, col_y = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
        cr = random.randint(4, 8)
        draw.ellipse([col_x - cr, col_y - cr, col_x + cr, col_y + cr], fill=(250, 240, 220))


def generate_scenarios(out_dir="sandbox/robustness_tests"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Touching Plates (2x3) ---
    # In a 3000x2000 img, each cell is 1000x1000. Radius 500 means they touch.
    img = Image.new("RGB", (3000, 2000), (15, 15, 20))
    draw = ImageDraw.Draw(img)
    r = 500
    for row in range(2):
        for col in range(3):
            create_plate(draw, col * 1000 + 500, row * 1000 + 500, r, f"T_{row}{col}")
    img.save(out_path / "case_touching.jpg")

    # --- 2. Random Offsets ---
    img = Image.new("RGB", (3000, 2000), (15, 15, 20))
    draw = ImageDraw.Draw(img)
    r = 300
    centers = [(400, 400), (1200, 600), (2500, 350), (600, 1500), (1600, 1400), (2200, 1700)]
    for i, (cx, cy) in enumerate(centers):
        create_plate(draw, cx, cy, r, f"R_{i}")
    img.save(out_path / "case_random_offsets.jpg")

    print(f"Generated robustness test set in {out_dir}")


if __name__ == "__main__":
    generate_scenarios()
