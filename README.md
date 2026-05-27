# Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields
<div align="center">
<h1>CVPR 2026</h1>

### [Webpage](https://ivl.cs.brown.edu/research/turbo-gs) | [arXiv](https://arxiv.org/abs/2412.13547) 

<img src="assets/teaser-figure-turbogs.png" alt="Turbo-GS Teaser" width="100%"/>

</div>

## Overview

Turbo-GS is a fast 3D Gaussian optimization pipeline for novel-view synthesis. It improves optimization efficiency while preserving rendering quality by combining:
- Dilation-based rendering support for high-resolution training (including 4K settings)
- Joint guidance from position and appearance errors for densification
- Convergence-aware budget control 


## Installation

### 1. Clone with submodules

```bash
git clone --recursive https://github.com/inspirelt/Turbo-GS.git
```

### 2. Create environment

```bash
conda env create -f environment.yml
conda activate turbo3dgs
```

Alternatively, follow [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install dependencies manually.

### 3. Build/install CUDA submodules

```bash
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim
```


## Data Format

Expected COLMAP layout:

```text
<scene>
|---images
|   |---0001.png
|   |---0002.png
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

You can prepare scenes using the provided conversion utility:

```bash
python convert.py -s <scene_path> [--resize]
```

## Training

Turbo-GS-style training controls commonly used in this repository:

```bash
python train.py \
  -s <scene_path> \
  -m <output_path> \
  --optimizer_type sparse_adam \
  --dilation 2 \
  --max_cap 1000000 \
  --color_ratio 0.8 \
  --ssim_ratio 1.0
```

Key flags:

- `--dilation`: dilation factor for training-time rendering
- `--max_cap`: max Gaussian budget cap (`-1` disables budget scheduler)
- `--color_ratio`: weighting for appearance-based densification guidance
- `--ssim_ratio`: weighting for SSIM contribution in the loss

## Rendering and Metrics

Render trained models:

```bash
python render.py -m <output_path>
```

Compute metrics:

```bash
python metrics.py -m <output_path>
```

## Citation

If this code or paper helps your research, please cite:

```bibtex
@misc{dhiman2025turbogsaccelerating3dgaussian,
      title={Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields}, 
      author={Ankit Dhiman and Tao Lu and R Srinath and Emre Arslan and Angela Xing and Yuanbo Xiangli and R Venkatesh Babu and Srinath Sridhar},
      year={2025},
      eprint={2412.13547},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13547}, 
}
```

## Acknowledgments

This project builds on top of the original 3D Gaussian Splatting codebase and related open-source contributions from the community.
