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

This branch contains Turbo-GS with the [Scaffold-GS](https://github.com/city-super/Scaffold-GS) backbone (Turbo-Scaffold-GS). The 3DGS version is on the [`release/3dgs`](https://github.com/inspirelt/Turbo-GS/tree/release/3dgs) branch.


## Installation

### 1. Clone

```bash
git clone https://github.com/inspirelt/Turbo-GS.git
```

### 2. Requirements

- Linux with an NVIDIA GPU and a driver that supports CUDA version for that GPU. 
- The CUDA version toolkit with `nvcc`, and `CUDA_HOME` pointing at it. `torch_scatter` and the three extensions in `submodules/` are compiled from source.
- replace the `cu130` wheels in `environment.yml` with your cuda version. 

### 3. Create environment and build the CUDA extensions

```bash
conda env create -f environment.yml
conda activate sgs_release
python -m pip install --no-build-isolation torch_scatter==2.1.2
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-dilated
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/fused-ssim
```

See [INSTALL.md](INSTALL.md) for the full sequence (environment prefix, cache directories, and a check that every extension imports). 


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

The Mip-NeRF 360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/).

## Training

### Mip-NeRF 360 (nine scenes)

`scripts/train_mip360.sh` trains the nine Mip-NeRF 360 scenes one after another, at native resolution (`-r 1 -i images --eval`):

```bash
bash scripts/train_mip360.sh <data_root> <output_root> [--preset 10k|15k|25k]
```

- `<data_root>`: the directory that holds the scene folders (`bicycle`, `bonsai`, `counter`, `flowers`, `garden`, `kitchen`, `room`, `stump`, `treehill`);
- `<output_root>`: each scene is written to `<output_root>/<scene>`;
- `--preset`: `10k`, `15k` (the default) or `25k`; see `arguments/presets.py`.

### Training a single scene

```bash
python train.py -s data/mipnerf360/garden -m outputs/garden --eval -i images -r 1
```

Key flags:

- `--preset`: `10k`, `15k` (default) or `25k` (see `arguments/presets.py`); any flag given on the command line overrides the preset
- `--dilation`: dilation factor for training-time rendering (`2` in every preset)
- `--max_anchors`: anchor budget cap (`-1` = no cap; each preset sets one)
- `--densify_color_threshold`: threshold for appearance-based densification (`-1` = off)
- `--data_device`: `cuda` (default) keeps the training images on the GPU; `cpu` keeps them on the host

## Rendering and Metrics

We've integrated the rendering and metrics calculation process into the training code. So, when completing training, the ```rendering results```, ```fps``` and ```quality metrics``` will be printed automatically. And the rendering results will be save in the log dir. Mind that the ```fps``` is roughly estimated by 

```
torch.cuda.synchronize();t_start=time.time()
rendering...
torch.cuda.synchronize();t_end=time.time()
```

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
@misc{turbogs,
      title={Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance Fields}, 
      author={Ankit Dhiman and Tao Lu and R Srinath and Emre Arslan and Angela Xing and Yuanbo Xiangli and R Venkatesh Babu and Srinath Sridhar},
      year={2025},
      eprint={2412.13547},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13547}, 
}
```

This branch is built on Scaffold-GS:

```bibtex
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={20654--20664},
  year={2024},
  organization={IEEE}
}
```

## License

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgments

This project builds on top of [Scaffold-GS](https://github.com/city-super/Scaffold-GS), the original [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) codebase, [fused-ssim](https://github.com/rahul-goel/fused-ssim) and related open-source contributions from the community.
