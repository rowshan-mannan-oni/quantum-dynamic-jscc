# Run cookbook

What to type to get each result. Every command assumes the environment is active and you are in
the repo root:

```bash
conda activate dyna_jscc
cd /path/to/quantum-dynamic-jscc
```

---

## The two rules

**1. There is no `--checkpoint path` argument.** Every script *rebuilds* the folder name from the
options you pass, so **evaluation must repeat the flags used for training**:

```bash
python train_dyna.py   --lambda_reward 2e-3 --seed 0 --quantum_site policy
python make_figures.py --lambda_reward 2e-3 --seed 0 --quantum_site policy   # same flags
```

Get one wrong and you get "file not found", not the wrong model — the name is the guard.

**2. On Linux, add `--num_workers 4`.** The default is 0 because Windows cannot spawn dataloader
workers here. On Linux that default leaves data loading as the bottleneck.

### Which flags decide the folder

| Flag | Effect on the folder name |
|---|---|
| `--C_channel` `--lambda_L2` `--lambda_reward` `--select` | `C16_L2_1.0_re_0.0015_hard` |
| `--quantum_site` `--n_qubits` `--vqc_layers` | `…_qpolicy-q8l2` |
| `--matched_site` `--n_qubits` | `…_mpolicy-q8` |
| `--seed` | `…_s0` |

Everything else (epochs, learning rates, batch size) does **not** appear, so re-running with a
different schedule overwrites the previous result. Change `--seed` or use a fresh
`--checkpoints_dir` if you want to keep both.

> **`--seed` is part of the name, so it is not optional at evaluation time.** The existing
> converted baseline was produced without one and lives in `C16_L2_1.0_re_0.0015_hard`, so
> evaluate it with **no** `--seed`; adding `--seed 0` looks for `…_hard_s0` and fails. Anything
> trained from now on with `--seed 0` needs `--seed 0` to evaluate.

---

## Before a long run: verify in ~2 minutes

```bash
# GPU visible, and the quantum stack actually backpropagates
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m quantum.smoke_test            # add --bench to time each site on your GPU

# the whole pipeline, 2 short epochs, into a throwaway folder
python train_dyna.py --gpu_ids 0 --lambda_reward 9.9e-3 \
    --n_epochs_joint 1 --n_epochs_decay 1 --n_epochs_fine 0 \
    --save_epoch_freq 1 --print_freq 25600
```

Worth the two minutes: a 400-epoch run that fails at hour three costs a day.

---

## Training

### Classical baseline — the reference

```bash
python train_dyna.py --gpu_ids 0 --select hard --SNR_MIN 0 --SNR_MAX 20 \
    --lambda_reward 1.5e-3 --seed 0 --num_workers 4
```

→ `Checkpoints/C16_L2_1.0_re_0.0015_hard_s0/` · ~3.4 h on an RTX 4070 · 400 epochs
(150 joint + 150 decay + 100 fine-tune)

### Quantum arm

```bash
python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed 0 --num_workers 4 \
    --quantum_site policy
```

→ `…_hard_qpolicy-q8l2_s0/` · ~5.3 h

Registered sites: `policy`. Others land as they are implemented — `python -c "import quantum;
print(quantum.available_sites())"` lists them. Add `--n_qubits 6` or `--vqc_layers 4` to vary the
circuit; both change the folder name.

### Matched control — the classical arm at the quantum parameter count

```bash
python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed 0 --num_workers 4 \
    --matched_site policy
```

→ `…_hard_mpolicy-q8_s0/` · ~3.4 h (classical speed)

Answers "was the difference the circuit, or just the smaller layer?". Without it, a quantum result
is ambiguous. `--quantum_site` and `--matched_site` cannot both be set.

### How often to checkpoint

| Flag | Unit | Default | Writes |
|---|---|---|---|
| `--save_latest_freq` | **images** | 20480 | overwrites `latest_*` plus optimizer state — **sets resume granularity** |
| `--save_epoch_freq` | **epochs** | 40 | numbered `{epoch}_net_*.pth` snapshots, weights only |

```bash
# resume point every epoch, numbered snapshot every 10
python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 \
    --save_latest_freq 49920 --save_epoch_freq 10
```

At `--batch_size 128` an epoch is 390 batches = **49,920 images**, so that is once per epoch; the
default fires about 2.4 times per epoch.

> **`--save_latest_freq` must be a multiple of `--batch_size`.** The test is
> `total_iters % save_latest_freq == 0` and `total_iters` advances by `batch_size`, so `20000` at
> batch 128 almost never fires. The default `20480` is exactly `160 × 128`.

> **Only `latest` can be resumed from.** Numbered snapshots skip the optimizer state, which is
> twice the size of the weights. `--save_epoch_freq` is for evaluation snapshots; lower
> `--save_latest_freq` for finer crash recovery.

Cost per save: weights 25.3 MB, optimizer state 50.5 MB.

| Situation | Setting |
|---|---|
| Kaggle, or anywhere a session can be cut short | `--save_latest_freq 24960` (twice an epoch) |
| Long local run, saving disk | `--save_epoch_freq 50` |
| Want a PSNR-against-epoch curve | `--save_epoch_freq 10` → 40 snapshots, about 1 GB |
| Debugging | `--save_epoch_freq 1` — 400 epochs is about **10 GB** |

`--save_by_iter` sends the periodic saves to `iter_<n>_net_*.pth` instead of overwriting `latest`.
It keeps every intermediate, but `latest` then only advances at `--save_epoch_freq` boundaries, so
a resume can lose more than you expect.

### Resume an interrupted run

```bash
python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed 0 --num_workers 4 \
    --continue_train
```

Repeat **every** flag from the original command, then add `--continue_train`. It reads the epoch,
optimizer state and Gumbel temperature from `latest` and carries on. Only `latest` can be resumed
from; numbered checkpoints carry weights only.

### Shorten a run

```bash
--n_epochs_joint 30 --n_epochs_decay 20 --n_epochs_fine 10     # 60 epochs
```

Fine for trends and debugging. **Not** for comparing arms unless both plateau — an under-trained
arm makes the comparison meaningless no matter how symmetric the protocol looks.

---

## Evaluation

### Scalar metrics at one SNR

```bash
# the existing converted baseline was made without a seed, so omit --seed here
python test_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 \
    --SNR 10 --num_test 10000 --num_test_channel 1
```

Prints mean PSNR, SSIM, transmitted groups, and the per-class breakdown. Runs one image at a time,
so keep `--num_test` modest for a quick look.

`--num_test_channel N` sends the same image through **N** independent channel draws and averages,
which smooths the noise at N× the cost. The default is 5; use 1 for a fast pass.

### All figures

```bash
python make_figures.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed 0 --num_workers 4 \
    --num_test 10000 --results_dir ./figures
```

→ `figures/` · ~10–20 min

| Output | |
|---|---|
| `fig4_rate_psnr_vs_snr.png` | rate (CPP) and PSNR against SNR |
| `fig5_attainable_psnr.png` | PSNR at each fixed rate, adaptive overlaid |
| `fig6_per_class.png` | per-class rate and PSNR |
| `reconstructions.png` | samples at several SNRs |
| `fig*_data.csv` | the numbers behind each figure |

Useful adjustments:

```bash
--snr_list 0,5,10,15,20      # coarser sweep, faster
--num_test 1000              # quick pass
--fig6_snr 5                 # per-class figure at another SNR
```

---

## Experiments

### Rate–distortion curve (one arm)

Each λ is a separate run and one point on the curve:

```bash
for L in 5e-4 7.5e-4 1e-3 1.5e-3 2.5e-3; do
    python train_dyna.py --gpu_ids 0 --lambda_reward $L --seed 0 --num_workers 4
    python make_figures.py --gpu_ids 0 --lambda_reward $L --seed 0 \
        --results_dir ./figures/lambda_$L
done
```

~17 h for five points. Higher λ penalises rate harder, so the model transmits less.

### Three-arm comparison at one λ

```bash
for ARM in "" "--matched_site policy" "--quantum_site policy"; do
  for S in 0 1 2; do
    python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed $S --num_workers 4 $ARM
    python make_figures.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed $S \
        --results_dir ./figures/arm$(echo $ARM | tr -d ' -') $ARM
  done
done
```

Nine runs, ~36 h. Three seeds per arm because a single run cannot separate a real effect from
run-to-run noise at these margins. Everything except the swapped module is held identical, which
is what makes the comparison fair rather than merely symmetric.

---

## Existing weights

Convert a single-file `{SE,CE,G,P}` checkpoint into the layout the scripts expect:

```bash
python convert_kaggle_weights.py --src Weight/final.pth --lambda_reward 1.5e-3 --force
```

Pass the same `--C_channel` / `--lambda_L2` / `--lambda_reward` / `--select` you will later pass to
`test_dyna.py` or `make_figures.py`, since they determine where it is written. Every state dict is
validated with `strict=True` before anything is saved.

> Folders created before the naming fix are called `C16_L2_1_re_<…>`. Rename them to
> `C16_L2_1.0_re_<…>` or nothing will find them.

---

## Kaggle

Upload `Kaggle_Train_and_Figures.ipynb`, set **Accelerator → GPU** and **Internet → On**, then edit
one cell:

```python
MODE = "train"        # or "weights" to use a checkpoint attached as a Dataset
QUICK_TEST = True     # False for the full 400-epoch schedule
LAMBDA_REWARD = 1.5e-3
```

Weights are not in the repo, so `MODE="weights"` needs your checkpoint attached via **Add Input**.
Never install the pinned `torch` from `requirements.txt` there — Kaggle's image already has a
working CUDA build.

---

## Reference

| | |
|---|---|
| `--gpu_ids 0` / `-1` | GPU 0 / CPU |
| `--select hard` \| `soft` | mask selection |
| `--lambda_reward` | rate penalty; **sweep this** for the tradeoff curve |
| `--SNR_MIN 0 --SNR_MAX 20` | training SNR range (dB) |
| `--batch_size 128` | paper setting |
| `--n_qubits 8` `--vqc_layers 2` | circuit width and depth; cost grows as 2^qubits |
| `--seed` | unset = unseeded; **set it for anything being compared** |
| `--checkpoints_dir ./Checkpoints` | move it to keep runs that would otherwise collide |
| `--print_freq` `--save_epoch_freq` | logging and snapshot frequency |

See [CHANGES.md](CHANGES.md) for what differs from the original code, and
[Quantum_JSCC_Plan.md](Quantum_JSCC_Plan.md) for why the experiments are shaped this way.
