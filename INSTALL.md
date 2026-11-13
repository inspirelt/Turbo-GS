# Installing Scaffold-GS-Turbo

## Requirements

- Linux with an NVIDIA GPU and a driver that supports CUDA version.
- **CUDA build tools:** the CUDA version toolkit with `nvcc`, and `CUDA_HOME` pointing at it.
  `torch_scatter` and the three extensions in `submodules/` are compiled from source during
  the install; without `nvcc` they fail to build. A host C++ compiler supported by that `nvcc`
  is needed too.
- **GPU memory:** about 50 GiB for native-4K Mip-NeRF 360 scenes with the default
  `--data_device cuda`, which keeps every training image on the GPU (peak 34–50 GiB in our measurements).
- conda (Miniconda is enough) and network access to `repo.anaconda.com`, `pypi.org` and
  `download.pytorch.org`.

## Commands

Run from the repository root. `ENV` is where the environment goes; the three cache variables
are optional and only keep downloads and build files off a small root disk.

Set `<env_dir>` to where the environment should go, `<cache_dir>` to any directory with a few GB free, and `CUDA_HOME` to the CUDA version toolkit (the directory that contains `bin/nvcc`) before running the commands.

```bash
ENV=<env_dir>
export CONDA_PKGS_DIRS=<cache_dir>/conda_pkgs
export PIP_CACHE_DIR=<cache_dir>/pip_cache
export TMPDIR=<cache_dir>/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
export PATH=$CUDA_HOME/bin:$PATH

conda env create -p "$ENV" -f environment.yml
"$ENV/bin/python" -m pip install --no-build-isolation torch_scatter==2.1.2
"$ENV/bin/python" -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-dilated
"$ENV/bin/python" -m pip install --no-build-isolation ./submodules/simple-knn
"$ENV/bin/python" -m pip install --no-build-isolation ./submodules/fused-ssim
"$ENV/bin/python" -c "import torch, torch_scatter, diff_gaussian_rasterization_dilated, simple_knn._C, fused_ssim; print(torch.__version__, torch.cuda.is_available())"
```

With a named environment instead of a prefix, use `conda env create -n sgs_release -f environment.yml`
and `conda run -n sgs_release python -m pip ...` (or activate it and call `python -m pip`).

`fused_ssim` in `submodules/fused-ssim` is upstream `rahul-goel/fused-ssim` at commit
`1272e21a282342e89537159e4bad508b19b34157`, the version every result here was trained with.

## Training

```bash
python train.py -s data/mipnerf360/garden -m outputs/garden --eval -i images -r 1
```

`--preset` selects `10k`, `15k` (default) or `25k` (see `arguments/presets.py`); any flag
given on the command line overrides the preset.
