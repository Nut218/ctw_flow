# CTW 12/3 + observation-residual conditional Flow (256-pixel example)

This folder is a portable **inference package** for the previously measured
43.46526391 dB CAVE 256×256 single-image result. It includes the exact Flow
checkpoint, shared CTW initialization, example LR-HSI/HR-MSI, SRF, inference
source, and configuration. Nothing outside this folder is required at runtime
apart from Python packages and a compatible PyTorch installation.

## Run

From inside this folder, after activating a compatible Python environment:

```powershell
python .\run.py
```

`run.py` also accepts an absolute path from any working directory and switches
to the package folder automatically. On the machine where this package was
verified, the command was:

```powershell
& 'F:\Anaconda\envs\myenv\python.exe' '.\run.py'
```

The output appears under `outputs/example_run/`. In its `run_config.json`,
the bundled example should give approximately PSNR 43.465 dB, SAM 4.198°, and
SSIM 0.97848 with the supplied checkpoint and initial factors. The standalone
verification measured 43.465920 dB, versus 43.465264 dB in the archived run.
Hardware,
library versions, and floating-point kernels may cause small differences.

For another `.mat` observation file, use:

```powershell
python run.py --data C:\path\to\observations.mat --srf C:\path\to\srf.mat --out_dir outputs\my_run
```

Custom data must contain `LRHSI` shaped `(H/4,W/4,31)` and `HRMSI` shaped
`(H,W,3)`; optional `HRHSI` is used **only for metrics**. The SRF `.mat` must
contain `srf` shaped `(3,31)`. The Flow checkpoint is tied to 31 HSI bands,
three MSI channels, scale 4, and TW ring/core ranks 12/3. Supplying `--data`
automatically disables the example's fixed initial factors so the run
initializes from your observations. Do not reuse the example's initial factors
for another scene.

## What is and is not bundled

- `run.py`, `run_ca_dps.py`, `ctw_ca/`: runnable CTW and guided Flow inference.
- `config.json`: original 20-step Flow and 300-step final joint optimization
  settings. The G2 holdout gate is enabled as in the recorded run, but its
  selected scale on the example was 0.
- `artifacts/flow_obscond_jointloss_light.pt`: pretrained observation-residual
  conditional Flow. It was trained on scene-separated 192×192 TW factor
  samples, using a light joint reconstruction/observation objective.
- `artifacts/shared_initial_factors.pt`: initialization for **only** the
  included example.
- `example/`: a sample input with a ground-truth HR-HSI for verification, the
  CAVE SRF, and the original result record.

The current source tree does **not** contain the original script that trained
this exact observation-residual-plus-joint-loss checkpoint. Its training
arguments are embedded in the `.pt` checkpoint, but this package does not
claim exact from-scratch retraining reproducibility. It reproduces inference
with the supplied weight. This is pretrained-prior-assisted test-time
self-supervised fusion, not an untrained zero-shot method.

## Dependencies

Verified on Windows with Python 3.9.21, PyTorch 2.1.0+cu121, NumPy 1.26.4,
SciPy 1.13.1, TensorLy 0.9.0, and scikit-image 0.24.0. Install PyTorch for
your hardware, then run `pip install -r requirements.txt`.
