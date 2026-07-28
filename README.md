# Quantum Dynamic JSCC

Hybrid classical–quantum deep Joint Source-Channel Coding (JSCC) for wireless image transmission with adaptive rate control.

This repository extends the Dynamic JSCC architecture of Yang & Kim with a **quantum variant**, implemented using [TorchQuantum](https://github.com/mit-han-lab/torchquantum), and provides a controlled comparison between the classical and quantum versions.

> **Status: work in progress.** The classical baseline is set up and reproducible. The quantum variant is under active development; no comparative results are published yet.

## Background

The base model performs deep JSCC over an AWGN channel with **adaptive rate control**: a policy network decides, per image and per channel SNR, how many latent channel groups to transmit — trading reconstruction quality against bandwidth. The architecture consists of:

| Module | File | Role |
|---|---|---|
| `netSE` — Source Encoder | `models/networks.py` | Downsampling CNN producing image features |
| `netCE` — Channel Encoder | `models/networks.py` | ResNet blocks + SNR modulation → transmitted latent |
| `netP` — Policy Network | `models/networks.py` | Gumbel-softmax gate selecting how many groups to send |
| `netG` — Generator / Decoder | `models/networks.py` | Reconstructs the image from the noisy latent |

## Research plan

The quantum variant replaces a selected classical module with a variational quantum circuit (`tq.QuantumModule`), following a hybrid pattern: `Linear (→ n_qubits) → angle encoding → variational circuit → measurement → Linear`.

Planned progression:

1. **Classical baseline** — fully trained reference under a fixed, logged protocol.
2. **Quantum policy network (`netP`)** — feasibility milestone. Small and self-contained, it validates end-to-end autograd through the quantum block and its interaction with the Gumbel-softmax decision.
3. **Quantum channel encoder (`netCE`)** — the site where the transmitted representation is formed, and the main research target.

### Comparison methodology

The goal is a **fair, controlled comparison**, not a claim of quantum advantage. Both arms are trained under an *identical* protocol — same data splits, channel model, SNR range, optimizer, epochs, batch size, and random seeds — with the quantum/classical block as the only difference. Parameter counts are matched where possible. Absolute numbers may differ from the original paper due to protocol differences; the contribution is the **relative** comparison, plus parameter-efficiency and convergence analysis.

## Setup

Requires an NVIDIA GPU with a recent driver. The original code targets a 2021 PyTorch release, which does **not** support modern GPUs (Ampere/Ada, e.g. RTX 3090/4070); the pinned versions below are current builds that run the original code unmodified.

```bash
conda create -n dyna_jscc python=3.10 -y
conda activate dyna_jscc

# CUDA build of PyTorch (adjust the CUDA tag to your driver if needed)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4
```

Or use the pinned file (note the CUDA index URL comment inside it):

```bash
pip install -r requirements.txt
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Platform note: `num_workers` on Windows

`train_dyna.py` and `test_dyna.py` execute all code at module top level with **no `if __name__ == '__main__':` guard**. On Linux this is fine (`fork`), but on Windows (`spawn`) each DataLoader worker re-imports the script and the run crashes with:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

This repo therefore sets `num_workers=0` in both scripts. **On Linux, raise it to `2`–`4`** for faster data loading — with CIFAR-10's small images, data loading is otherwise the bottleneck.

### Dataset

CIFAR-10 downloads automatically to `./data` on first run. The upstream source (`cs.toronto.edu`) can be very slow; if so, fetch `cifar-10-python.tar.gz` manually and place it in `./data/` — torchvision will verify its MD5 and extract it.

## Usage

Settings live in `options/base_options.py`, `options/train_options.py`, and `options/test_options.py`. The option-parsing style is borrowed from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Training

```bash
python train_dyna.py --gpu_ids '0' --select 'hard' --SNR_MIN 0 --SNR_MAX 20 --lambda_reward 2e-3
```

A full run is **400 epochs** (`n_epochs_joint 150` + `n_epochs_decay 150` + `n_epochs_fine 100`). Checkpoints are written to `./Checkpoints/<config-name>/`.

Key options:

| Flag | Meaning |
|---|---|
| `--select` | `hard` or `soft` mask selection |
| `--lambda_reward` | Weight of the rate penalty — **sweep this to trace the rate–distortion curve** |
| `--C_channel` | Latent channel count |
| `--G_n` / `--G_s` | Non-selective / selective group counts |
| `--SNR_MIN`, `--SNR_MAX` | Training SNR range (dB) |

### Testing

```bash
python test_dyna.py --gpu_ids '0' --select 'hard' --SNR_MIN 0 --SNR_MAX 20 \
    --lambda_reward 2e-3 --num_test 10000 --num_test_channel 1 --SNR 5
```

Reports mean PSNR and mean transmitted-channel count. Requires a trained checkpoint.

### Figures

`test_dyna.py` only prints scalars. To produce the paper's figures from a trained checkpoint:

```bash
python make_figures.py --gpu_ids 0 --select hard --lambda_reward 1.5e-3 --num_test 10000
```

This writes to `./figures/` (override with `--results_dir`):

| Output | Contents |
|---|---|
| `fig4_rate_psnr_vs_snr.png` | Average rate (CPP) and PSNR against SNR |
| `fig5_attainable_psnr.png` | PSNR at each fixed rate, with adaptive points overlaid |
| `fig6_per_class.png` | Per-class rate and PSNR at one SNR |
| `reconstructions.png` | Sample reconstructions at several SNRs |
| `fig4_data.csv`, `fig5_data.csv`, `fig6_data.csv` | The numbers behind each figure |

The fixed-rate curves in `fig5` come from `DynaAWGNModel.forced_active`, which pins the rate
instead of letting the policy network choose.

### Using existing weights

`convert_kaggle_weights.py` converts a single-file checkpoint that packs all four
sub-networks as `{'SE', 'CE', 'G', 'P'}` into the per-network layout `base_model.py` expects.
Each state dict is validated with `strict=True` before anything is written:

```bash
python convert_kaggle_weights.py --src Weight/final.pth --lambda_reward 1.5e-3
```

Note that `test_dyna.py` and `make_figures.py` locate a checkpoint by *rebuilding the folder
name* from `--C_channel`, `--lambda_L2`, `--lambda_reward` and `--select`, so pass the same
values to the converter and to whatever consumes its output.

### Running on Kaggle

`Kaggle_Figures.ipynb` clones this repository and runs the scripts above, so the figures always
come from the latest commit rather than a copy of the code pasted into a notebook. It needs
Internet enabled (to clone) and your checkpoint attached as a Kaggle Dataset — weights are not
stored in the repository. Do **not** install the pinned `torch` from `requirements.txt` on
Kaggle; its image already ships a working CUDA build.

### Reproducing the rate–distortion curves

The paper's tradeoff curves are traced by sweeping `--lambda_reward` (each value is one operating point), for both `hard` and `soft` selection. Because the model is *dynamic*, a single trained model already yields a spread of operating points.

## Attribution

This work builds directly on:

- **Original repository:** https://github.com/mingyuyng/Dynamic_JSCC (Mingyu Yang)
- **Paper:** [Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control](https://arxiv.org/abs/2110.04456) — Mingyu Yang, Hun-Seok Kim
- Portions of the code carry a **Copyright (C) 2017 NVIDIA Corporation** notice and originate from the CycleGAN/pix2pix codebase.

```bibtex
@misc{yang2021deep,
  title={Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control}, 
  author={Mingyu Yang and Hun-Seok Kim},
  year={2021},
  eprint={2110.04456},
  archivePrefix={arXiv},
  primaryClass={eess.SP}
}
```

## License

Licensed under **CC BY-NC-SA 4.0**, inherited from the upstream project — see [LICENSE](LICENSE).

This means derivative works must be shared under the same license (**ShareAlike**), **non-commercial use only**, and the original authors must be credited (**Attribution**). Original license headers in the source files are retained.