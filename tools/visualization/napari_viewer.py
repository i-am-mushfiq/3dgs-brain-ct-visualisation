import os
import sys
import argparse

os.environ['NAPARI_ASYNC'] = '0'

import numpy as np
import imageio.v2 as imageio
import napari
from natsort import natsorted
from glob import glob
from qtpy.QtCore import QTimer

# Command-line argument parser
parser = argparse.ArgumentParser(description="3D Volume Viewer with optional auto-rotation.")
parser.add_argument('--rotation', choices=['on', 'off'], default='off', help='Enable or disable auto-rotation')
args = parser.parse_args()

auto_rotate_enabled = args.rotation == 'on'

# Load images
image_folder = 'Brain2_PNG'
files = natsorted(glob(f"{image_folder}/*.png"))
if not files:
    print(f"❌ No PNG files found in {image_folder}")
    sys.exit(1)

slices = [imageio.imread(f) for f in files]
volume = np.stack(slices, axis=0)

# Setup Napari viewer
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(
    volume,
    name='Brain Volume',
    rendering='attenuated_mip',
    attenuation=0.003,
    colormap='twilight',
)

# Rotation setup
rotation_speed_deg_per_tick = 1.0

def rotate_camera():
    if auto_rotate_enabled:
        pitch, yaw, roll = viewer.camera.angles
        yaw = (yaw + rotation_speed_deg_per_tick) % 360
        viewer.camera.angles = (pitch, yaw, roll)

# Timer for smooth updates
timer = QTimer()
timer.setInterval(50)  # ms
timer.timeout.connect(rotate_camera)
timer.start()

napari.run()
