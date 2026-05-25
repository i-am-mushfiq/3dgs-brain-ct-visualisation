# convert_jp2_to_tif.py
# A script to decode JP2 images to TIFF using OpenJPEG's opj_decompress.exe

import sys
import subprocess
from pathlib import Path

def convert_folder(input_dir, output_dir, openjpeg_bin):
    """
    Convert all .jp2 images in input_dir to .tif in output_dir
    using the opj_decompress executable.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for jp2_file in input_dir.glob('*.jp2'):
        output_tif = output_dir / f"{jp2_file.stem}.tif"
        try:
            subprocess.run([
                openjpeg_bin,
                '-i', str(jp2_file),
                '-o', str(output_tif)
            ], check=True)
            print(f"✅ Converted {jp2_file.name} → {output_tif.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed for {jp2_file.name}: {e}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_jp2_to_tif.py <input_folder> <output_folder> <path_to_opj_decompress>")
        sys.exit(1)

    src_folder = sys.argv[1]
    dst_folder = sys.argv[2]
    opj_bin = sys.argv[3]

    if not Path(opj_bin).is_file():
        print(f"Error: opj_decompress executable not found at {opj_bin}")
        sys.exit(1)

    convert_folder(src_folder, dst_folder, opj_bin)

if __name__ == '__main__':
    main()
