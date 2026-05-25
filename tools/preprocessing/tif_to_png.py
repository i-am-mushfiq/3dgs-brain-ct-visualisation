from PIL import Image
import os

INPUT_DIR = 'Output_Data'
OUTPUT_DIR = 'Output_PNG'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith('.tif'):
        tif_path = os.path.join(INPUT_DIR, filename)
        png_name = os.path.splitext(filename)[0] + '.png'
        png_path = os.path.join(OUTPUT_DIR, png_name)

        try:
            img = Image.open(tif_path)
            img.save(png_path)
            print(f'✅ {filename} → {png_name}')
        except Exception as e:
            print(f'❌ Failed {filename}: {e}')
