# GEN3C Setup with Pixi

This document describes how to set up and run the [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/) pipeline using [Pixi](https://pixi.sh/) for environment management, replacing the original Conda-based workflow.

## What is GEN3C?

GEN3C is NVIDIA's generative video model (CVPR 2025 Highlight) for producing 3D-consistent video with precise camera control. It is built on top of the Cosmos 7B video diffusion model and supports three input modes: single image, dynamic video, and multi-view images.

## What is Pixi?

Pixi is a modern, cross-platform package manager by [prefix.dev](https://prefix.dev/) that serves as a replacement for Conda and venv. Key advantages:

- **Single manifest file** (`pixi.toml`) — manages both Conda and PyPI dependencies in one place
- **Lockfile-based** (`pixi.lock`) — guarantees fully reproducible environments
- **No manual activation** — run commands via `pixi run <cmd>` or define reusable tasks
- **Fast resolver** — significantly faster dependency solving compared to Conda

## Prerequisites

- **OS:** Linux (tested on Ubuntu 20.04, 22.04, 24.04)
- **GPU:** NVIDIA H100 or A100 (minimum ~43GB VRAM with full offloading)
- **Pixi:** Install with `curl -fsSL https://pixi.sh/install.sh | bash`
- **HuggingFace account:** Required to download model weights (free read-access token)

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/nv-tlabs/GEN3C.git
cd GEN3C
```

### 2. Initialize Pixi from the Conda environment file

Pixi can directly import the existing Conda YAML file, which bootstraps the `pixi.toml` with the correct Conda-channel dependencies (Python 3.10, CUDA 12.4, GCC 12.4, etc.):

```bash
pixi init --import cosmos-predict1.yaml
```

This creates a `pixi.toml` with the Conda dependencies already configured. The PyPI dependencies from `requirements.txt` need to be added manually to the `[pypi-dependencies]` section (see the provided `pixi.toml` in this repository for the complete configuration).

### 3. Install all dependencies

```bash
pixi install
```

This resolves and installs both Conda and PyPI packages in a single command.

### 4. Fix NVIDIA include symlinks

Same as the original Conda workflow — required for building Transformer Engine:

```bash
pixi run bash -c '
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
'
```

### 5. Install additional compiled packages

These packages require building from source or special installation and cannot be declared in `pixi.toml`:

```bash
# Transformer Engine
pixi run pip install transformer-engine[pytorch]==1.12.0

# NVIDIA Apex (built from source with CUDA extensions)
git clone https://github.com/NVIDIA/apex
pixi run bash -c 'CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex'

# MoGe (Microsoft monocular geometry estimator)
pixi run pip install git+https://github.com/microsoft/MoGe.git
```

### 6. Verify the environment

```bash
pixi run test-env
```

Expected output:

```
[SUCCESS] torch found
[SUCCESS] torchvision found
[SUCCESS] diffusers found
[SUCCESS] transformers found
[SUCCESS] megatron.core found
[SUCCESS] transformer_engine found
-----------------------------------------------------------
[SUCCESS] Cosmos environment setup is successful!
```

### 7. Access to HuggingFace repos

GEN3C's default download script pulls several gated NVIDIA/Meta models. Before running the download, accept each license by clicking "Agree and access repository" on these pages:

| Repo | Needed for | Approval |
|---|---|---|
| `nvidia/Gen3C-Cosmos-7B` | Main diffusion model (required) | Instant |
| `nvidia/Cosmos-Tokenize1-CV8x8x8-720p` | Video tokenizer (required) | Instant |
| `nvidia/Cosmos-Guardrail1` | Content safety filter (optional) | Instant |
| `meta-llama/Llama-Guard-3-8B` | Prompt safety filter (optional) | Manual review by Meta |
| `google-t5/t5-11b` | Text encoder for prompt conditioning (optional, ~42GB) | Open |

**Minimal setup:** We only need the first two. The guardrail models and T5 can be skipped at inference time with flags.

### 8. Download model checkpoints

Log in to HuggingFace with a read-access token (generate one at https://huggingface.co/settings/tokens):

```bash
pixi run bash -c 'huggingface-cli login'
```

**Option A — Minimal download (~29GB, only the two required models):**

```bash
pixi run bash -c 'huggingface-cli download nvidia/Gen3C-Cosmos-7B --local-dir checkpoints/Gen3C-Cosmos-7B --include "*.pt" "*.json" README.md'
pixi run bash -c 'huggingface-cli download nvidia/Cosmos-Tokenize1-CV8x8x8-720p --local-dir checkpoints/Cosmos-Tokenize1-CV8x8x8-720p --include "*.pt" "*.jit" "*.json" README.md'
```

**Option B — Full download (~90GB, everything including guardrails and T5):**

```bash
pixi run download-checkpoints
```

### 9. Run inference on a sample image

**Minimal inference (skips guardrails, T5 text encoder, and prompt upsampler):**

```bash
pixi run gen3c-single-image -- \
  --input_image_path assets/diffusion/000000.png \
  --video_save_name test_single_image \
  --disable_guardrail --disable_prompt_encoder --disable_prompt_upsampler
```

The three disable flags are what make the minimal download setup work — without them the pipeline would try to load the T5 text encoder and the guardrail models.

**Full inference (with text conditioning and guardrails, requires Option B download):**

```bash
pixi run gen3c-single-image -- \
  --input_image_path assets/diffusion/000000.png \
  --video_save_name my_output
```

The minimal setup was successfully tested on a single NVIDIA H100 80GB:

- **Input:** `assets/diffusion/000000.png`
- **Output:** 121-frame video at `outputs/test_single_image.mp4`
- **Total runtime:** ~8 minutes (model loading: ~2 min, diffusion: ~6 min)
- **Command:**
  ```bash
  pixi run gen3c-single-image -- \
    --input_image_path assets/diffusion/000000.png \
    --video_save_name test_single_image \
    --disable_guardrail --disable_prompt_encoder --disable_prompt_upsampler
  ```

## Pixi Tasks

The `pixi.toml` defines convenience tasks that set the required environment variables automatically:

| Task | Command | Description |
|---|---|---|
| `pixi run test-env` | Runs `scripts/test_environment.py` | Verify all critical imports |
| `pixi run download-checkpoints` | Runs `scripts/download_gen3c_checkpoints.py` | Download model weights from HuggingFace |
| `pixi run gen3c-single-image` | Runs `gen3c_single_image.py` | Generate video from a single image |

Each task automatically sets `LD_LIBRARY_PATH`, `CUDA_HOME`, and `PYTHONPATH`.

## Challenges Encountered and Resolutions

### 1. `setuptools` version conflict between Conda and PyPI

**Problem:** The original `requirements.txt` pins `setuptools==76.0.0`, but Pixi's Conda solver installed `setuptools==82.0.1` as part of the base environment. Since Pixi treats Conda-installed packages as constraints for PyPI resolution, the exact pin `==76.0.0` was unsatisfiable.

**Resolution:** Relaxed the version constraint to `setuptools >= 76.0.0` in the `[pypi-dependencies]` section of `pixi.toml`. This is safe because setuptools is a build tool and the minor version difference does not affect GEN3C's runtime behavior.

### 2. `transformer_engine` fails to find `libnvrtc` at import time

**Problem:** When importing `transformer_engine`, it runs `ldconfig -p | grep 'libnvrtc'` to locate the NVRTC library. On systems where CUDA is installed via Conda/Pixi (rather than system-wide), `ldconfig` does not know about the libraries inside the Pixi environment, so the grep returns nothing and the import crashes with a `CalledProcessError`.

**Resolution:** Set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` before any Python command that imports `transformer_engine` or `megatron.core`. This is baked into all the Pixi task definitions in `pixi.toml` so users do not need to remember it.

### 3. `transformer_engine` version mismatch assertion (specific to the HPC cluster I am using)

**Problem:** On the HPC clusters, a system-wide pip configuration (`/cvmfs/soft.computecanada.ca/config/python/pip-x86-64-v3-gentoo2023.conf`) intercepts all `pip install` commands and substitutes custom-patched wheels from a local wheelhouse. These wheels have a local version suffix. The `transformer_engine` package performs a strict version string comparison across its three sub-packages (`transformer-engine`, `transformer-engine-cu12`, `transformer-engine-torch`), and the mixed version strings cause an `AssertionError`.

**Resolution:** Two steps:
1. Downloaded the vanilla `transformer-engine==1.12.0` wheel directly from PyPI using `PIP_CONFIG_FILE=/dev/null` to bypass the ComputeCanada pip config, then installed it with `--no-deps --force-reinstall`.
2. For `transformer-engine-cu12` (which is not published on PyPI as a standalone wheel), patched the installed dist-info metadata to remove the `+computecanada` suffix from the version string by renaming the `.dist-info` directory and editing the `METADATA` and `RECORD` files.

**Note:** This issue is specific to the HPC clusters I was using. On standard systems, `pip install transformer-engine[pytorch]==1.12.0` should work without any workaround.

### 4. MoGe pulls incompatible `huggingface_hub` version

**Problem:** Installing MoGe from its Git repository (`pip install git+https://github.com/microsoft/MoGe.git`) pulled in `huggingface_hub>=1.11.0` as a transitive dependency (via `gradio`). This broke `transformers==4.49.0` and `tokenizers==0.21.4`, which both require `huggingface_hub<1.0`.

**Resolution:** After installing MoGe, downgraded huggingface-hub back to the pinned version:

```bash
pixi run pip install huggingface-hub==0.29.2
```

The resulting conflict warning for `gradio` is harmless — gradio is only used for MoGe's optional web UI, which GEN3C does not use.

### 5. NVIDIA Apex must be built from source

**Problem:** Apex is not available as a pre-built PyPI wheel with CUDA extensions. It must be compiled from source against the environment's CUDA toolkit, which means it cannot be declared as a dependency in `pixi.toml`.

**Resolution:** Cloned the Apex repository and built it inside the Pixi environment with the correct `CUDA_HOME` path:

```bash
CUDA_HOME=$CONDA_PREFIX pip install -v --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./apex
```

This is a manual post-install step that users must run after `pixi install`.

## Multi-View Video Input Support — Investigation

**GEN3C (the `Gen3C-Cosmos-7B` model) does *not* support multi-view video input in the current version of code.** It supports multi-view *images* through `gen3c_multiview.py`, but there is no code in the GEN3C pipeline that accepts multiple synchronized videos from different cameras as conditioning.

A separate Cosmos model bundled in the same repository (`Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview`) *does* generate multi-view *output* video for autonomous-driving scenarios, but it is a different model trained for a different task and is not GEN3C.


### What each GEN3C inference script supports

I traced the data shapes through each GEN3C inference script to understand exactly what input is accepted.

| Script | Input shape | Dimensions | Interpretation |
|---|---|---|---|
| [`gen3c_single_image.py`](cosmos_predict1/diffusion/inference/gen3c_single_image.py) | `(1, C, H, W)` | 1 view × 1 frame | Single image |
| [`gen3c_dynamic.py`](cosmos_predict1/diffusion/inference/gen3c_dynamic.py) | `(T, C, H, W)` | 1 view × T frames | Single-camera video |
| [`gen3c_multiview.py`](cosmos_predict1/diffusion/inference/gen3c_multiview.py) | `(N, C, H, W)` | N views × 1 frame | Sparse multi-view images |

Evidence for the multi-view script (from [gen3c_multiview.py:181](cosmos_predict1/diffusion/inference/gen3c_multiview.py#L181)):

```python
images_key = torch.tensor(npz["images_key_frames"], ...)  # (N, C, H, W), [-1,1]
depth_key  = torch.tensor(npz["depth_key_frames"],  ...)  # (N, 1, H, W)
K_key      = torch.tensor(npz["K_key_frames"],      ...)  # (N, 3, 3)
w2c_key    = torch.tensor(npz["w2cs_key_frames"],   ...)  # (N, 4, 4)
```

Each of the N keyframes is a single RGB image with its own intrinsics and pose. There is no time axis alongside the view axis — the tensor shape is `(N_views, C, H, W)`, not `(N_views, T_frames, C, H, W)`. So a "multi-view video" with, say, 5 cameras capturing 30 frames each (shape `(5, 30, C, H, W)`) cannot be expressed in this format.

### Related multi-view capability in the Cosmos family

The repository also ships [`video2world_multiview.py`](cosmos_predict1/diffusion/inference/video2world_multiview.py), which uses a different model checkpoint (`Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview`). It takes a seed video plus per-view text prompts and **generates** multi-view video for autonomous-driving rigs:

```python
parser.add_argument("--prompt_left",  ...)   # left camera view
parser.add_argument("--prompt_right", ...)   # right camera view
parser.add_argument("--prompt_back",  ...)   # rear camera view
parser.add_argument("--prompt_back_left",  ...)
parser.add_argument("--prompt_back_right", ...)
```

This is distinct from GEN3C in three ways:
1. Different model weights (`Cosmos-Predict1-7B-Video2World-Sample-AV-Multiview`, not `Gen3C-Cosmos-7B`)
2. Different pipeline class (`DiffusionVideo2WorldMultiviewGenerationPipeline`, not `Gen3cPipeline`)
3. Specialized for driving scenes with a fixed rig layout — not general-purpose multi-view video synthesis

