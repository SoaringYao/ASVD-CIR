# Training-free Acceleration for Blind Compressed Image Restoration via Activation-Calibrated Singular Value Decomposition

Suhang Yao, Xin Li, Xijun Wang, Yufei Wang, Yeying Jin, and Zhibo Chen

**Accepted for publication in IEEE JETCAS (2026)** · [IEEE Xplore](https://doi.org/10.1109/JETCAS.2026.3745896) · [arXiv](https://arxiv.org/abs/2610.29959)

ASVD uses input and output activation magnitudes to compute low-rank factors for pretrained linear and convolutional layers. Calibration and factorization run offline, with no retraining. This repository includes a PyTorch implementation and a [PromptCIR example](#promptcir-example) for blind JPEG restoration.

## Installation

Python 3.10 or later and PyTorch 2.0 or later are required.

```bash
python -m pip install .
```

## Usage

```python
import torch
from asvd import compress, load, save

device = torch.device("cuda")
model = build_model().to(device).eval()
model.load_state_dict(torch.load("original.pth", map_location="cpu", weights_only=True))

manifest = compress(model, rank=64,
                    calibration=(x.to(device) for x in calibration_inputs),
                    min_saving=0.25)
save(model, manifest, "compressed.pt")

model = build_model().to(device).eval()
load(model, "compressed.pt")
with torch.inference_mode():
    output = model(input_tensor.to(device))
```

Supply `build_model`, `calibration_inputs`, and `input_tensor` from your application. `compress` replaces eligible child layers in place. It supports standard `nn.Linear` and `nn.Conv2d` with `groups=1` and `padding_mode="zeros"`, each registered under a single name. Use `selector(name, layer)` to restrict the selection. The requested positive integer rank must reduce each selected layer’s weight count by at least `min_saving` (in `[0, 1)`). The checkpoint includes the manifest and model state. Load it into the same base architecture on the desired device and dtype.

Use representative calibration inputs that exercise every selected layer. The default call is `model(batch)`. For tuple or dictionary batches, pass `forward=lambda m, b: m(*b)` or `forward=lambda m, b: m(**b)`. The channel statistics and SVD use FP32. Actual inference latency depends on layer shapes, dtype, and hardware.

## PromptCIR example

From the ASVD source directory, clone PromptCIR and install the example dependencies. Download `model_best.pth` to `weights/model_best.pth`. [PromptCIR source](https://github.com/lbc12345/PromptCIR-NTIRE24) · [Pretrained weights](https://drive.google.com/drive/folders/1oxOVDK845rikuwWiMICHUv5nFGJjA7pV?usp=drive_link)

```bash
git clone https://github.com/lbc12345/PromptCIR-NTIRE24.git PromptCIR
python -m pip install '.[promptcir]'

python -m examples.promptcir compress \
  --checkpoint weights/model_best.pth \
  --calibration /path/to/natural_images \
  --output weights/promptcir_asvd.pt

python -m examples.promptcir infer \
  --checkpoint weights/promptcir_asvd.pt \
  --input /path/to/compressed.jpg \
  --output restored.png
```

The example generates calibration JPEGs with quality factors from 10 to 70 and saves one compressed PromptCIR model at rank 64 by default. Input images must be at least 32 pixels tall and wide. Use `--device cpu` or `--model-file /path/to/model.py` before the subcommand to override the default `cuda` device or `PromptCIR/model.py` path. Add `--cuda-graph` to `infer` to replay fixed-size tiles on CUDA. For PyTorch 2.0, use `numpy<2` for image conversion.

## Citation

```bibtex
@article{yao2026asvd,
  title   = {Training-free Acceleration for Blind Compressed Image Restoration via Activation-Calibrated Singular Value Decomposition},
  author  = {Yao, Suhang and Li, Xin and Wang, Xijun and Wang, Yufei and Jin, Yeying and Chen, Zhibo},
  journal = {IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
  year    = {2026},
  doi     = {10.1109/JETCAS.2026.3745896}
}
```

For work using the example backbone, also cite Li et al., *PromptCIR: Blind Compressed Image Restoration with Prompt Learning*, CVPR Workshops 2024. [PromptCIR paper](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Li_PromptCIR_Blind_Compressed_Image_Restoration_with_Prompt_Learning_CVPRW_2024_paper.pdf)

ASVD is distributed under the [Apache 2.0 license](LICENSE).
