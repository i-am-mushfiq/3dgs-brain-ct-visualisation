# Thesis Project — Status Report
**Date:** 2026-05-25  
**Project:** 3D Gaussian Splatting for Cinematic Anatomy on Consumer-Class Devices  
**Stage:** Pre-Defence  

---

## 1. Project Overview

This thesis implements and extends the *Cinematic Gaussians* method (Niedermayr et al., 2024 — TU Munich & Siemens Healthineers) to enable interactive, photorealistic visualization of medical anatomy on consumer-class devices (mobile, VR). The core technique is compressed 3D Gaussian splatting applied to real anatomical datasets (human brain MRI at 18 µm resolution).

---

## 2. Repository at a Glance

| Metric | Value |
|---|---|
| Total tracked + local size | ~12.7 GB |
| Total file count (excl. git/venv) | ~18,476 files |
| Primary implementation | `cinematic-gaussians/` (~11.1 GB) |
| Preprocessing tools | `JP2 to PNG converter/` (~1.2 GB) |
| 3D viewer | `Napari Viewer(Not a must have)/` (~1.4 GB) |
| Medical imaging source data | `Brain1_JP2/` + `Brain2_JP2/` (~136 MB tracked) |

---

## 3. Component Inventory

### 3.1 Core Implementation — `cinematic-gaussians/`

| File / Folder | Purpose | Status |
|---|---|---|
| `train.py` | Main training loop (L1 + SSIM loss, TensorBoard logging, multi-GPU) | Complete |
| `compress.py` | Importance-score compression + vector quantization pipeline | Complete |
| `export_ply.py` | Exports trained Gaussians to PLY format | Complete |
| `make_cameras.py` | Camera setup utilities | Complete |
| `environment.yml` | Conda env — CUDA 12.1, PyTorch 2.5.1, Python 3.12 | Up to date |
| `(deprecated)environment.yml` | Old environment file | Dead — should be deleted |
| `dummy.txt` | Placeholder | Stale — should be deleted |
| `gaussian_me/` | Core Python package (model, renderer, optimizer, IO, utils) | Complete |
| `submodules/` | CUDA rasterizer + KNN/NN (diff-gaussian-rasterization, simple-knn, simple-nn) | Present |
| `docs/` | Project webpage with static assets, sample videos | Present |
| `Rendering_Related_Files/` | Rendering scripts + 281 rendered PNGs + 1 MP4 output | Functional |

### 3.2 Gaussian Model Package — `gaussian_me/`

| Module | Role |
|---|---|
| `model.py` | `GaussianModel` — stores xyz, spherical harmonics, scale, rotation, opacity |
| `renderer.py` | CUDA rasterization via `GaussianRasterizer` |
| `optim.py` | Learning rate scheduling per parameter group |
| `compression.py` | Quantization and bit-packing logic |
| `eval.py` | PSNR / SSIM metrics |
| `args.py` | Argument parser — `ModelParams`, `OptimizationParams`, `CompressionParams` |
| `io/colmap_loader.py` | Loads COLMAP camera poses and point clouds |
| `io/dataset_readers.py` | Scene info, camera info, dataset readers |
| `utils/` | Camera transforms, SH utilities, loss functions, graphics math |

### 3.3 Rendering Tools — `Rendering_Related_Files/`

| File | Purpose |
|---|---|
| `Test_Renderer.py` | Loads model, renders single image |
| `preview.py` | Animated matplotlib preview over all views |
| `render_png_sequence.py` | Batch PNG frame export |
| `render_video.py` | MP4 output from rendered frames |
| `cinematic_splat.mp4` | 39.7 MB sample video output |
| `render_test.png` | 2.2 MB sample render |
| `video_frames/` | 281 rendered PNG frames (~2–3 MB each, ~700 MB total) |

### 3.4 Data Preprocessing Tools

| Tool | Purpose | Status |
|---|---|---|
| `JP2 to PNG converter/full_converter.py` | OpenJPEG wrapper — JP2 → TIF | Functional |
| `JP2 to PNG converter/tif_to_png.py` | TIF → PNG conversion | Functional |
| `Napari Viewer(Not a must have)/view_3d.py` | Interactive 3D volume viewer with auto-rotation | Functional, optional |

### 3.5 Medical Imaging Source Data

| Dataset | Format | Resolution | Notes |
|---|---|---|---|
| `Brain1_JP2/` | JP2 slices | 18 µm (LADAF-2021-17) | 100+ slices, tracked in git (~79 MB) |
| `Brain2_JP2/` | JP2 slices | High-res | Tracked in git (~58 MB) |
| `cinematic-gaussians/Brain2_PNG/` | Converted PNG | — | Output of conversion pipeline |

---

## 4. Experiment Log

### 4.1 Brain Dataset Experiments (Primary)

| Model | Size on Disk | Notes |
|---|---|---|
| `Test_Model/` | 596 MB | Earliest run — large, likely uncompressed |
| `Test_Model_v2/` | 47 MB | Compressed |
| `Test_Model_v3/` | 53 MB | — |
| `Test_Model_v4/` | 35 MB | — |
| `Test_Model_v5/` | 31 MB | — |
| `Test_Model_v6/` | 74 MB | — |
| `Test_Model_v7/` | 60 MB | Latest brain model, checkpoints at 100–1000 iter |

### 4.2 Tank (Baseline) Dataset Experiments

| Model | Size on Disk | Notes |
|---|---|---|
| `Tank_Test_Model_v7/` | 312 MB | — |
| `Tank_Test_Model_v8/` | 91 MB | — |
| `Tank_Test_Model_v9/` | 1.09 GB | Extensive checkpoints (100–1000 iterations saved) |
| `Tank_Test_Model_v10/` | ~0 MB | Empty / abandoned run |

### 4.3 Training Data Versions

| Folder | Size | Notes |
|---|---|---|
| `Training_Data/` | 289 MB | Original |
| `Training_Data_v2/` | 183 MB | — |
| `Training_Data_v3/` | 422 MB | — |
| `Training_Data_v4/` | 422 MB | Latest; v3 and v4 identical size — may be duplicates |

---

## 5. Current State Assessment

### What Is Working
- Complete end-to-end pipeline: data ingest → preprocessing → training → compression → rendering → video export
- Core `gaussian_me/` package is modular and well-structured
- Compression pipeline reduces models from ~600 MB to ~30–75 MB
- Rendering produces high-quality output (281 frames, 1 MP4)
- Two brain datasets ingested and converted from JP2
- COLMAP-based camera calibration integrated

### What Is Incomplete / Missing
- **No thesis document** — no `.pdf`, `.docx`, or `.tex` file found anywhere in the project root
- **No top-level README** — the only README is inside `cinematic-gaussians/` (the original paper's readme)
- **No experiment log or results table** — no structured record of hyperparameters, PSNR/SSIM scores, or run configurations across model versions
- **No pipeline documentation** — no guide describing how to run the full workflow end-to-end
- **`Tank_Test_Model_v10/`** is empty (129 KB) — abandoned run, should be removed
- **`(deprecated)environment.yml`** and **`dummy.txt`** are stale artifacts

### Risk Areas
- `Training_Data_v3/` and `Training_Data_v4/` are the same size (422 MB) — likely duplicated data consuming ~422 MB unnecessarily
- `Tank_Test_Model_v9/` alone is 1.09 GB with checkpoints at every 100 iterations — unclear if all are needed
- 281 rendered PNGs (~700 MB) stored inside the implementation repo with no clear purpose after the MP4 was produced
- All large folders are gitignored locally — no backup or remote storage strategy visible

---

## 6. Git History Summary

| Commit | Message |
|---|---|
| `b5a0cbc` | Phase 2 — Thesis_v1.2: brain data pipeline, tools, and code improvements |
| `8c9b1a5` | Phase 1 — Thesis: initial cinematic-gaussians exploration |

Only 2 commits on `master`. The `cinematic-gaussians/` subdirectory has its own inner `.git` repo tracking that component separately.

---

## 7. Suggested Better Folder Structure

> **No changes made.** This is a proposal only.

### Problems with Current Layout
1. Implementation code, training data, model outputs, and tools are all mixed in a flat structure
2. No separation between the thesis *document* and the thesis *code*
3. Tool folders have informal names (`Napari Viewer(Not a must have)`)
4. No results / evaluation layer — metrics live only in TensorBoard logs
5. Multiple versioned model outputs have no metadata (what changed, what the PSNR was)
6. Source data (JP2 files) sits at the same level as code tools

### Proposed Structure

```
Thesis_PreDefence/
│
├── docs/                          ← Thesis writing
│   ├── thesis.tex / thesis.docx   ← Main thesis document
│   ├── figures/                   ← Diagrams, renders used in thesis
│   └── references.bib
│
├── src/                           ← All implementation code
│   ├── cinematic-gaussians/       ← Core 3DGS implementation (current cinematic-gaussians/)
│   │   ├── gaussian_me/           (unchanged)
│   │   ├── submodules/            (unchanged)
│   │   ├── train.py
│   │   ├── compress.py
│   │   ├── export_ply.py
│   │   ├── make_cameras.py
│   │   └── environment.yml
│   │
│   ├── preprocessing/             ← Data conversion tools
│   │   ├── jp2_to_png.py          (merged from JP2 to PNG converter/)
│   │   └── tif_to_png.py
│   │
│   └── visualization/             ← Optional viewing tools
│       └── napari_viewer.py       (from Napari Viewer/)
│
├── data/                          ← All datasets (gitignored except metadata)
│   ├── raw/                       ← Original source data
│   │   ├── Brain1_JP2/            (current Brain1_JP2/)
│   │   └── Brain2_JP2/            (current Brain2_JP2/)
│   │
│   └── processed/                 ← Converted/prepared data
│       ├── brain_v1/              ← Training_Data/ → named after subject/version
│       ├── brain_v4/              ← Training_Data_v4/ (latest only)
│       └── tank/                  ← Tank dataset
│
├── experiments/                   ← All runs, clearly named
│   ├── brain/
│   │   ├── run_001_baseline/      ← Test_Model/ (with metadata.json)
│   │   ├── run_007_final/         ← Test_Model_v7/ (best brain model)
│   │   └── ...
│   │
│   └── tank/
│       ├── run_007/               ← Tank_Test_Model_v7/
│       ├── run_009_best/          ← Tank_Test_Model_v9/ (largest, most checkpoints)
│       └── ...
│
│   (each run folder contains a metadata.json):
│   ├── metadata.json              ← {date, iterations, PSNR, SSIM, config used}
│   └── point_cloud/               ← Model checkpoints
│
├── results/                       ← Final outputs for thesis
│   ├── renders/                   ← Final render frames (video_frames/ → here)
│   ├── cinematic_splat.mp4        ← Final video
│   ├── metrics.csv                ← Consolidated PSNR/SSIM table across all runs
│   └── figures/                   ← Plots for thesis
│
├── report.md                      ← This file
├── README.md                      ← Top-level project README (missing — needs creation)
├── .gitignore                     ← (keep, extend to cover data/ and experiments/)
└── environment.yml                ← Top-level env (or pointer to src/cinematic-gaussians/)
```

### Key Changes Explained

| Change | Reason |
|---|---|
| `docs/` at root | Thesis writing belongs at the top level, not buried or absent |
| `src/` wrapping all code | Clear separation between code and data/outputs |
| `JP2 to PNG converter/` → `src/preprocessing/` | Removes informal folder name with spaces |
| `Napari Viewer(Not a must have)/` → `src/visualization/` | Removes parenthetical uncertainty from folder name |
| `data/raw/` + `data/processed/` | Separates source from derived data; both gitignored |
| `experiments/` with `metadata.json` per run | Makes version history legible without digging into folder sizes |
| `results/` | Single place for thesis-facing outputs (renders, metrics, figures) |
| Drop `Training_Data/`, `v2`, `v3` | Only latest (`v4`) needed; others are intermediate artifacts |
| Drop `Test_Model/` (v1–v6) | Only best-performing model kept; rest archived or deleted |
| Drop `Tank_Test_Model_v10/` | Empty — no value |
| Drop `(deprecated)environment.yml`, `dummy.txt` | Stale files |

---

*Report generated: 2026-05-25*
