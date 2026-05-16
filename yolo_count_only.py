import os
import sys
import csv
import argparse
import subprocess
from pathlib import Path

# --- BOOTSTRAP: Auto-Install Dependencies ---
def bootstrap():
    try:
        import ultralytics
    except ImportError:
        print("Required library 'ultralytics' not found. Installing now...")
        try:
            # We use 'ultralytics' as the main package
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
            print("Installation successful!\n")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)

# Run bootstrap before anything else
bootstrap()
from ultralytics import YOLO

def run_counter(input_path, model_path, output_dir):
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure you have the 'colony_model.pt' file in the 'models' folder.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 2. Prepare Output Directory
    out_path = Path(output_dir).resolve()
    img_out_path = out_path / "annotated_images"
    img_out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "colony_counts.csv"
    
    # 3. Identify Input Files
    input_p = Path(input_path)
    if input_p.is_dir():
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in extensions:
            image_files.extend(list(input_p.glob(ext)))
        
        # Remove duplicates (happens on case-insensitive filesystems)
        image_files = sorted(list(set(image_files)))
    else:
        image_files = [input_p]

    if not image_files:
        print(f"No valid images found at {input_path}")
        return

    print(f"Found {len(image_files)} images. Starting count...\n")

    # 4. Process Images One-by-One
    file_exists = csv_file.exists()
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Filename', 'Colony_Count'])

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for img_path in image_files:
            if img_path.stat().st_size == 0:
                print(f" [SKIP] {img_path.name} (Empty file)")
                continue

            # Perform inference
            results = model.predict(
                source=str(img_path),
                imgsz=1280,
                conf=0.15,
                iou=0.8,
                max_det=1000,
                save=False,
                verbose=False
            )
            
            result = results[0]
            count = len(result.boxes)
            original_name = img_path.name
            
            # Save the annotated image
            save_path = img_out_path / original_name
            result.save(filename=str(save_path), labels=False, conf=False)
            
            # Log to CSV
            writer.writerow([now, original_name, count])
            print(f" [OK] {original_name}: {count} colonies")

    print(f"\nProcessing Complete!")
    print(f"Annotated images: {img_out_path}")
    print(f"CSV log:          {csv_file}")

if __name__ == "__main__":
    # We use -in and -out for clarity as requested
    parser = argparse.ArgumentParser(description="Professional Colony Counter CLI", prefix_chars='-')
    parser.add_argument("-in", required=True, help="Path to input image or folder of images")
    parser.add_argument("-out", required=True, help="Path to output results directory")
    parser.add_argument("-model", default="models/colony_model.pt", help="Path to model file (default: models/colony_model.pt)")
    
    # Simple fix for argparse interpretation of -in and -out
    args = parser.parse_args()
    
    # Access arguments using the names defined above
    run_counter(getattr(args, 'in'), args.model, args.out)
