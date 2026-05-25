
# Thesis(PreDefence) — 3D Gaussian Splatting for Cinematic Anatomy on Consumer-Class Devices

**Pre-defence research repository**

This repository contains the complete implementation, preprocessing toolchain, and experimental artefacts for a thesis investigating the application of compressed 3D Gaussian splatting to interactive cinematic anatomy visualisation on consumer-class hardware.

The work builds on the *Cinematic Gaussians* method (Niedermayr et al., 2024 — TU Munich & Siemens Healthineers) and extends it to handle **orthographic volumetric medical imaging data**, specifically high-resolution human brain CT stacks acquired via phase-contrast synchrotron tomography.

---

## Research Context

Interactive photorealistic visualisation of anatomical structures — commonly referred to as *Cinematic Anatomy* — has historically required high-end GPU workstations and high-bandwidth storage. Novel view synthesis via 3D Gaussian splatting (3DGS) offers a promising pathway to compress such representations into lightweight, hardware-agnostic formats suitable for mobile devices and VR environments.

This thesis addresses a gap in the existing literature: the upstream 3DGS pipeline was designed exclusively for **perspective-camera scenes** (photographic or synthetic). Medical volumetric datasets are inherently **orthographic** — CT and MRI stacks are parallel projections of physical tissue sections with calibrated inter-slice spacing. Adapting the pipeline to this geometry, and validating it on real anatomical data at sub-20-µm resolution, constitutes the core technical contribution.

---

## Repository Structure

```
Thesis_PreDefence/
│
├── cinematic-gaussians/          3DGS implementation (git submodule → fork)
│   ├── gaussian_me/              Core Python package (model, renderer, optimiser, IO)
│   ├── make_cameras.py           ★ Orthographic camera generator (thesis contribution)
│   ├── export_ply.py             ★ Trained model → PLY exporter (thesis contribution)
│   ├── train.py                  Training loop (Windows multiprocessing fix applied)
│   ├── compress.py               Vector-quantisation compression pipeline
│   ├── environment.yml           Full conda lock file (Python 3.12, PyTorch 2.5.1, CUDA 12.1)
│   └── CONTRIBUTION.md           Itemised diff vs. upstream
│
├── tools/
│   ├── preprocessing/
│   │   ├── jp2_to_tif.py         ★ JP2 → TIF batch converter (OpenJPEG wrapper)
│   │   └── tif_to_png.py         ★ TIF → PNG batch converter (Pillow)
│   └── visualization/
│       ├── napari_viewer.py      ★ Interactive 3D volume viewer with auto-rotation
│       └── requirements.txt      napari, imageio, natsort, qtpy
│
├── data/
│   └── raw/                      Source JP2 datasets (gitignored — large binary)
│       ├── Brain1_JP2/           18.048 µm voxel pitch — high-res VOI (LADAF-2021-17)
│       └── Brain2_JP2/           202.0 µm voxel pitch — whole-organ scan
│
├── assets/
│   └── reference.png             Reference screenshot
│
├── report.md                     Project status and structure analysis
├── README.md                     This file
└── .gitmodules                   Submodule pointer to i-am-mushfiq/cinematic-gaussians
```

> ★ marks files written from scratch as part of this thesis.

---

## Contributions at a Glance

| Contribution | File | What it does |
|---|---|---|
| Orthographic camera generator | `cinematic-gaussians/make_cameras.py` | Converts a PNG slice stack into a `cameras.json` encoding parallel-projection geometry, enabling the 3DGS pipeline to train on CT/MRI data without COLMAP |
| PLY exporter | `cinematic-gaussians/export_ply.py` | Exports a trained `.npz` checkpoint to PLY format for inspection in MeshLab / CloudCompare |
| Dataset reader fixes | `cinematic-gaussians/gaussian_me/io/dataset_readers.py` | Rewrote `fetchPly()` for robust field handling; fixed tuple unpacking in `readNerfSyntheticInfo()`; added `readVolumeSceneInfo()` for the volume camera format |
| Windows multiprocessing fix | `cinematic-gaussians/train.py` | Replaced unpicklable lambda `collate_fn` with named `collate_identity()` — required for `DataLoader` with `num_workers > 0` on Windows (spawn-based multiprocessing) |
| Full conda lock file | `cinematic-gaussians/environment.yml` | Pinned dependency graph for exact reproducibility across machines |
| JP2 → TIF converter | `tools/preprocessing/jp2_to_tif.py` | CLI wrapper around `opj_decompress` for batch decoding of JPEG 2000 medical images |
| TIF → PNG converter | `tools/preprocessing/tif_to_png.py` | Batch TIF → PNG conversion via Pillow, producing training-ready slices |
| 3D volume viewer | `tools/visualization/napari_viewer.py` | Napari-based interactive 3D viewer with attenuated MIP rendering and optional auto-rotation, for qualitative inspection of the brain stack prior to training |

The upstream code (`cinematic-gaussians/`) is tracked as a git submodule pointing to a personal fork. All thesis-specific changes are isolated to a single, reviewable commit:  
[i-am-mushfiq/cinematic-gaussians — compare vs. upstream](https://github.com/i-am-mushfiq/cinematic-gaussians/compare/KeKsBoTer:cinematic-gaussians:master...master)

---

## Dataset

**LADAF-2021-17** — Human brain specimen, from the [LADAF project](https://ladaf.univ-lille.fr/), Université de Lille.

| Dataset | Resolution | Modality | Notes |
|---|---|---|---|
| Brain1 (`Brain1_JP2/`) | 18.048 µm/voxel | Phase-contrast synchrotron CT | Regional VOI — sub-voxel anatomical detail |
| Brain2 (`Brain2_JP2/`) | 202.0 µm/voxel | Phase-contrast synchrotron CT | Whole-organ scan |

Raw data is not redistributed here. Access the original dataset through the LADAF project. Converted PNG slices are produced locally by the preprocessing pipeline and are gitignored.

---

## Prerequisites

### Hardware
- NVIDIA GPU with CUDA 12.1 support (training and rendering)
- Minimum 8 GB VRAM recommended; 24 GB for high-resolution runs without threshold tuning

### Software
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [OpenJPEG](https://www.openjpeg.org/) — `opj_decompress` binary required for JP2 conversion
- Git with submodule support

---

## Setup

### 1. Clone with submodule

```bash
git clone https://github.com/i-am-mushfiq/Thesis_PreDefence.git
cd Thesis_PreDefence
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
cd cinematic-gaussians
conda env create -f environment.yml
conda activate cin3dgs
```

### 3. Build CUDA extensions

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/simple-nn
```

### 4. Install visualization dependencies (optional)

```bash
cd ../tools/visualization
pip install -r requirements.txt
```

---

## Full Pipeline Walkthrough

### Stage 1 — Ingest raw JP2 data

Raw brain scans arrive as JPEG 2000 files in `data/raw/Brain1_JP2/` or `data/raw/Brain2_JP2/`. Decode to TIFF first:

```bash
python tools/preprocessing/jp2_to_tif.py \
    data/raw/Brain1_JP2/ \
    /tmp/brain1_tif/ \
    /path/to/opj_decompress.exe
```

Convert TIFF to PNG (edit `INPUT_DIR` / `OUTPUT_DIR` inside the script):

```bash
# Edit tools/preprocessing/tif_to_png.py:
#   INPUT_DIR  = '/tmp/brain1_tif/'
#   OUTPUT_DIR = 'cinematic-gaussians/Training_Data_v4/images/'

python tools/preprocessing/tif_to_png.py
```

### Stage 2 — Inspect the volume (optional)

Before training, verify the converted slices look correct using the 3D viewer:

```bash
# Edit image_folder inside napari_viewer.py to point at your PNG directory
python tools/visualization/napari_viewer.py --rotation on
```

The viewer loads the stack as a 3D volume with attenuated maximum intensity projection and renders it interactively. `--rotation on` enables automatic yaw rotation for overview inspection.

### Stage 3 — Generate orthographic cameras

```bash
# Edit IMAGE_DIR, OUTPUT_JSON, and SLICE_THICK inside make_cameras.py
cd cinematic-gaussians
python make_cameras.py
```

Key parameters:

| Parameter | Value used | Meaning |
|---|---|---|
| `SLICE_THICK` | `0.02525` mm | Physical inter-slice spacing — from scanner metadata |
| `ORTHO_FOVY` | `1e-4` rad | Near-zero FOV approximates true orthographic projection |

Output: `cameras.json` in the scene folder, consumed directly by `train.py`.

### Stage 4 — Train

```bash
cd cinematic-gaussians
conda activate cin3dgs

python train.py \
    -s Training_Data_v4 \
    -m Test_Model_v7 \
    --eval \
    --test_iterations 7000 15000 30000 \
    --densify_grad_threshold 0.00005 \
    --save_iterations 30000
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir Test_Model_v7
```

**VRAM tuning:** Increase `--densify_grad_threshold` (e.g. `0.0001`) to cap the Gaussian count if you run out of memory.

### Stage 5 — Compress

```bash
python compress.py \
    -m Test_Model_v7 \
    --eval \
    --output_vq Test_Model_v7_compressed \
    --load_iteration 30000
```

The compression script reports PSNR and SSIM on train and test splits and writes a compressed `.npz` checkpoint. Typical output size: 30–75 MB (down from 400–600 MB uncompressed).

### Stage 6 — Export to PLY (optional)

```bash
# Edit npz_path and out_ply inside export_ply.py
python export_ply.py
```

Open the resulting `.ply` in MeshLab or CloudCompare to inspect Gaussian density and spatial distribution over anatomical structures.

### Stage 7 — Render

```bash
# Single image
python Rendering_Related_Files/Test_Renderer.py

# Animated matplotlib preview over all views
python Rendering_Related_Files/preview.py

# Full PNG sequence (one file per view)
python Rendering_Related_Files/render_png_sequence.py

# Encode to MP4
python Rendering_Related_Files/render_video.py
```

---

## Experimental Runs

All model outputs are gitignored (large binary). The following runs were conducted locally:

### Brain dataset (primary)

| Run | Output folder | Size | Notes |
|---|---|---|---|
| Baseline | `Test_Model/` | 596 MB | Uncompressed, early hyperparameters |
| v2 | `Test_Model_v2/` | 47 MB | First successful compression |
| v5 | `Test_Model_v5/` | 31 MB | Smallest model |
| v7 | `Test_Model_v7/` | 60 MB | Best quality–size trade-off; used for final renders |

### Tank dataset (baseline validation)

| Run | Output folder | Size | Notes |
|---|---|---|---|
| v7 | `Tank_Test_Model_v7/` | 312 MB | Initial tank experiment |
| v9 | `Tank_Test_Model_v9/` | 1.09 GB | Dense checkpointing (every 100 iterations) |

### Training data versions

| Folder | Size | Notes |
|---|---|---|
| `Training_Data/` | 289 MB | Original preparation |
| `Training_Data_v4/` | 422 MB | Final preparation used for v7 model |

---

## Upstream Reference

This project is based on:

> Simon Niedermayr, Christoph Neuhauser, Kaloian Petkov, Klaus Engel, Rüdiger Westermann.  
> *Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices.*  
> arXiv:2404.11285, 2024.  
> [Project page](https://keksboter.github.io/cinematic-gaussians/) · [arXiv](https://arxiv.org/abs/2404.11285) · [Upstream code](https://github.com/KeKsBoTer/cinematic-gaussians)

```bibtex
@misc{niedermayr2024novel,
    title         = {Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices},
    author        = {Simon Niedermayr and Christoph Neuhauser and Kaloian Petkov and Klaus Engel and Rüdiger Westermann},
    year          = {2024},
    eprint        = {2404.11285},
    archivePrefix = {arXiv},
    primaryClass  = {cs.GR}
}
```
