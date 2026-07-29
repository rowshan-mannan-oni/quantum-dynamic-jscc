# Run cookbook

What to type to get each result, and what every flag does.

```bash
conda activate dyna_jscc
cd /path/to/quantum-dynamic-jscc
```

All commands assume that. On Linux add `--num_workers 4` to anything that reads data.

**Contents** — [rules](#the-two-rules) · [training arguments](#training-arguments) ·
[inference arguments](#inference-arguments) · [runs by site](#runs-by-site) ·
[checkpointing](#checkpointing-and-keeping-the-best) · [evaluation](#evaluation) ·
[experiments](#experiments)

---

## The two rules

**1. There is no `--checkpoint path` argument.** Every script *rebuilds* the folder name from the
options you pass, so **evaluation must repeat the flags used for training**:

```bash
python train_dyna.py   --lambda_reward 2e-3 --seed 0 --quantum_site policy
python make_figures.py --lambda_reward 2e-3 --seed 0 --quantum_site policy   # same flags
```

Get one wrong and you get "file not found", not the wrong model — the name is the guard.

**2. `--num_workers` defaults to 0**, because the scripts run at module scope with no
`__main__` guard and Windows cannot spawn dataloader workers into that. On Linux, **use 2–4** or
data loading becomes the bottleneck.

### Which flags decide the folder

| Flag | Contribution to the name |
|---|---|
| `--C_channel` `--lambda_L2` `--lambda_reward` `--select` | `C16_L2_1.0_re_0.0015_hard` |
| `--quantum_site` `--n_qubits` `--vqc_layers` | `…_qpolicy-q8l2` |
| `--matched_site` `--n_qubits` | `…_mpolicy-q8` |
| `--seed` | `…_s0` |

Nothing else appears — **epochs, learning rates and batch size do not**, so re-running with a
different schedule silently overwrites the previous result. Vary `--seed` or point
`--checkpoints_dir` somewhere else to keep both.

> `--seed` is part of the name, so it is not optional afterwards. A run trained with `--seed 0`
> needs `--seed 0` to evaluate; one trained without a seed must be evaluated without one.

---

## Training arguments

### Choosing the arm

| Flag | Default | What it does |
|---|---|---|
| `--quantum_site` | `none` | Module(s) to replace with a variational circuit, comma separated. An unknown name aborts rather than falling back. |
| `--matched_site` | `none` | Module(s) to replace with a *classical* control of the same shape and parameter budget. Mutually exclusive with `--quantum_site`. |
| `--n_qubits` | `8` | Circuit width, and the bottleneck width of the matched control. Cost grows as 2^n. |
| `--vqc_layers` | `2` | Circuit depth. **Deeper is not safer** — random deep circuits have exponentially vanishing gradients. |

Leaving both site flags at `none` gives the unmodified classical model.

### Identifying the experiment

| Flag | Default | What it does |
|---|---|---|
| `--lambda_reward` | `1.5e-3` | Weight on the rate penalty. **Sweep this** to trace the rate–distortion curve; higher transmits less. |
| `--lambda_L2` | `1.0` | Weight on reconstruction MSE. |
| `--select` | `hard` | `hard` or `soft` mask selection during training. Inference always uses the hard mask. |
| `--seed` | unset | Seeds torch, numpy and random. **Set it for anything being compared**, and vary it across repeats to separate a real effect from run-to-run noise. |
| `--checkpoints_dir` | `./Checkpoints` | Where runs are written. |

### The channel

| Flag | Default | What it does |
|---|---|---|
| `--SNR_MIN` `--SNR_MAX` | `0` `20` | Training SNR range in dB, sampled uniformly per image. At evaluation the SNR is fixed by `--SNR` instead. |

### Schedule

| Flag | Default | What it does |
|---|---|---|
| `--n_epochs_joint` | `150` | Stage 1: everything trains at `--lr_joint`. |
| `--n_epochs_decay` | `150` | Stage 2: same modules at the lower `--lr_decay`. |
| `--n_epochs_fine` | `100` | Stage 3: `--lr_fine`, **with `netSE` and `netP` frozen**. |
| `--lr_joint` `--lr_decay` `--lr_fine` | `5e-4` `5e-5` `1e-5` | Learning rate for each stage. |
| `--batch_size` | `128` | Paper setting. 390 batches per epoch on the full training set. |
| `--temp_init` | `5` | Initial Gumbel-Softmax temperature. |
| `--eta` | `0.015` | Per-epoch temperature decay, `temp *= exp(-eta)`, floored at 0.005. |

The stage is derived from the epoch, so it is correct when resuming into the middle of a run.

> The fine-tuning stage freezes `netSE` and `netP`. A quantum module at the `policy` or
> `source_encoder` site therefore stops training for the last 100 epochs. It applies equally to
> every arm, so the comparison stays fair — but check the loss curve before concluding a quantum
> arm converged.

### Checkpointing

| Flag | Default | What it does |
|---|---|---|
| `--save_epoch_freq` | `5` | Overwrite `latest_*` every N epochs. Also always written at the final epoch. |
| `--save_latest_freq` | `20480` | Same, part way through an epoch, counted in **images**. Must be a multiple of `--batch_size`. `0` disables. |
| `--snapshot_freq` | `0` | Additionally keep numbered `{epoch}_net_*.pth`. No optimizer state, so for evaluation curves rather than resuming. ~25 MB each. |
| `--val_size` | `0` | Images held out of training to score the model for `best_*`. `0` scores the epoch's training loss instead. |
| `--val_freq` | `1` | Score every N epochs. |
| `--continue_train` | off | Resume: load weights, optimizer, epoch and temperature. |
| `--epoch` | `latest` | Which tag to resume from — `latest` or `best`. |
| `--epoch_count` | `1` | Starting epoch when not resuming. Overridden by `--continue_train`. |

### Runtime

| Flag | Default | What it does |
|---|---|---|
| `--gpu_ids` | `0` | GPU index, comma separated for several, `-1` for CPU. |
| `--num_workers` | `0` | Dataloader workers. **2–4 on Linux**, must stay 0 on Windows. |
| `--print_freq` | `10240` | Log losses every N images. |
| `--verbose` | off | Print the full network architectures at startup. |

### Architecture — leave these alone

Changing any of them makes a run incomparable to everything else, and only `--C_channel` is
recorded in the folder name.

`--ngf 64` `--max_ngf 256` `--n_downsample 2` `--n_blocks 2` `--C_channel 16` `--G_n 4` `--G_s 4`
`--input_nc 3` `--output_nc 3` `--norm batch` `--init_type normal` `--init_gain 0.02`
`--model DynaAWGN`

`--C_channel 16` over an 8×8 grid is 1024 values, split into `G_n + G_s = 8` groups of 128, giving
the five rates CPP ∈ {0.25, 0.313, 0.375, 0.438, 0.5}.

### Inherited flags that do nothing

`--ndf`, `--max_dataset_size` and `--phase` are left over from the CycleGAN codebase this was
forked from and are never read. `--suffix` appends to the folder name; `--load_iter` and
`--save_by_iter` work but interact badly with resuming.

---

## Inference arguments

Shared by `test_dyna.py` and `make_figures.py`. **Every flag from the "identifying the experiment"
and "choosing the arm" tables above must be repeated**, because they locate the checkpoint.

| Flag | Default | What it does |
|---|---|---|
| `--epoch` | `latest` | Which checkpoint to load — `latest`, `best`, or a snapshot number. |
| `--SNR` | `5` | Channel SNR in dB (`test_dyna.py` only). |
| `--num_test` | `10000` | How many test images. The full CIFAR-10 test split is 10000. |
| `--num_test_channel` | `5` | Independent channel draws per image, averaged. Smooths noise at N× the cost; use 1 for a fast pass. |

`make_figures.py` adds:

| Flag | Default | What it does |
|---|---|---|
| `--snr_list` | `0,2,…,20` | SNRs to sweep for the rate/PSNR figures. |
| `--fig6_snr` | `10` | SNR used for the per-class figure. |
| `--eval_batch` | `250` | Batch size for evaluation. |
| `--results_dir` | `./figures` | Where figures and CSVs are written. |
| `--dataroot` | `./data` | Where CIFAR-10 lives. |

---

## Runs by site

### The classical reference — shared by every site

Train this **once**. It is the reference for all comparisons; you do not retrain it per site.

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000
```

→ `C16_L2_1.0_re_0.0015_hard_s0/` · ~3.4 h on an RTX 4070

> Whatever you choose for `--val_size` and `--seed` here **must be identical on every other arm**,
> or the arms are not trained on the same data.

### Site A — policy network ✅ implemented

Replaces the MLP mapping pooled features + SNR to one logit per rate.
Classical 21,253 params · matched 2,181 · quantum 2,141.

```bash
# matched control
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --matched_site policy

# quantum
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site policy
```

→ `…_hard_mpolicy-q8_s0/` ~3.4 h · `…_hard_qpolicy-q8l2_s0/` ~5.3 h

### Site B — channel encoder projection ⬜ not yet implemented

Replaces `Conv3×3(256→16)`, whose output **is** the transmitted signal. The headline site.
Classical 36,880 params · matched ~2,272 · quantum ~2,232. Circuit sees batch 8192.

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --matched_site projection

python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site projection
```

→ `…_hard_mprojection-q8_s0/` ~3.4 h · `…_hard_qprojection-q8l2_s0/` ~6.7 h

### Site C — decoder input ⬜ not yet implemented

Replaces `netG.mask_conv`, `Conv3×3(16→256)`. The receive-side mirror of B; pair with B for a
"quantum link".

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site decoder

# both ends at once
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site projection,decoder
```

→ `…_hard_qdecoder-q8l2_s0/` ~6.7 h · `…_hard_qprojection+decoder-q8l2_s0/` ~10 h

### Site D — SNR modulation ⬜ not yet implemented

Replaces the four `modulation` blocks that scale and shift features from the SNR.

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site modulation
```

→ `…_hard_qmodulation-q8l2_s0/` ~5.3 h

### Site F — source encoder ⬜ not yet implemented

Replaces the whole of `netSE` with a 4×4 quantum patch embedding: 378,944 params → ~2,728.

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --quantum_site source_encoder
```

→ `…_hard_qsource_encoder-q8l2_s0/` ~7.3 h

> The circuit emits 8 numbers per patch but the module outputs 256 channels, so every patch is
> described by 8 numbers regardless. Expect the largest quality drop here, and read it as evidence
> about *where* a circuit belongs rather than as a failure.

### Which sites exist right now

```bash
python -c "import quantum; print(quantum.available_sites())"
```

---

## Checkpointing and keeping the best

A run writes **two** sets of weights:

| Set | Written | Contains |
|---|---|---|
| `latest_*` | every `--save_epoch_freq` epochs, and at the final epoch | newest model, overwritten |
| `best_*` | whenever the score improves | best model the run reached |

```bash
python train_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 \
    --save_epoch_freq 5 --save_latest_freq 0 --val_size 2000
```

Evaluate either with `--epoch`:

```bash
python make_figures.py --gpu_ids 0 --lambda_reward 1.5e-3 --epoch best
python test_dyna.py    --gpu_ids 0 --lambda_reward 1.5e-3 --epoch latest
```

### What "best" means

The score is the **training objective**, `lambda_L2 · MSE + lambda_reward · groups` — not PSNR
alone, so a model is not called better for reconstructing well while transmitting more than it
should. Scoring runs at a fixed SNR grid with a fixed noise seed so epochs are comparable, and the
RNG state is restored afterwards so scoring does not perturb training.

`--val_size 0` scores the epoch's training loss instead: free, but it tracks the objective being
optimised rather than generalisation.

> `--val_size 2000` trains on 48,000 images rather than 50,000. Use the same value on every arm.

Best-tracking earns its keep on the quantum arms: a run does not necessarily end at its best point,
and variational circuits can destabilise late in training.

### Save frequency

> `--save_latest_freq` counts **images** and must be a multiple of `--batch_size`, or it rarely
> fires. The default `20480` is exactly `160 × 128`; one epoch is `49,920`.

> Only `latest` and `best` can be resumed from — snapshots carry no optimizer state.

Cost per save: weights 25.3 MB, optimizer state 50.5 MB.

| Situation | Setting |
|---|---|
| Session could be cut short | `--save_latest_freq 24960` (twice an epoch) |
| Disk is tight | `--save_epoch_freq 10 --save_latest_freq 0` |
| Want a PSNR-against-epoch curve | `--snapshot_freq 10` → 40 snapshots, ~1 GB |

### Resume

```bash
python train_dyna.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --val_size 2000 --continue_train
```

Repeat **every** flag from the original command, then add `--continue_train`. It resumes from the
epoch after the one `latest` recorded, so you lose up to `--save_epoch_freq − 1` epochs.
`--epoch best --continue_train` rewinds to the best point instead, discarding what came after.

### Shorten a run

```bash
--n_epochs_joint 30 --n_epochs_decay 20 --n_epochs_fine 10     # 60 epochs
```

Fine for trends and debugging. **Not** for comparing arms unless both have plateaued.

---

## Evaluation

### Scalar metrics at one SNR

```bash
python test_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --seed 0 \
    --SNR 10 --num_test 10000 --num_test_channel 1
```

Prints mean PSNR, SSIM, transmitted groups, and the per-class breakdown. One image at a time, so
keep `--num_test` modest for a quick look.

### All figures

```bash
python make_figures.py --gpu_ids 0 --num_workers 4 \
    --lambda_reward 1.5e-3 --seed 0 --num_test 10000 --results_dir ./figures
```

→ ~10–20 min

| Output | |
|---|---|
| `fig4_rate_psnr_vs_snr.png` | rate (CPP) and PSNR against SNR |
| `fig5_attainable_psnr.png` | PSNR at each fixed rate, adaptive overlaid |
| `fig6_per_class.png` | per-class rate and PSNR |
| `reconstructions.png` | samples at several SNRs |
| `fig*_data.csv` | the numbers behind each figure |

```bash
--snr_list 0,5,10,15,20      # coarser sweep, faster
--num_test 1000              # quick pass
--fig6_snr 5                 # per-class figure at another SNR
```

---

## Experiments

### Before a long run — verify in ~2 minutes

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m quantum.smoke_test            # --bench times each site, --gpu N picks a card

python train_dyna.py --gpu_ids 0 --lambda_reward 9.9e-3 \
    --n_epochs_joint 1 --n_epochs_decay 1 --n_epochs_fine 0 \
    --save_epoch_freq 1 --print_freq 25600
```

A 400-epoch run that fails at hour three costs a day.

### Three-arm comparison at one site

```bash
SITE=policy
COMMON="--gpu_ids 0 --num_workers 4 --lambda_reward 1.5e-3 --val_size 2000"

for S in 0 1 2; do
  for ARM in "" "--matched_site $SITE" "--quantum_site $SITE"; do
    python train_dyna.py $COMMON --seed $S $ARM
    python make_figures.py $COMMON --seed $S $ARM \
        --results_dir ./figures/${SITE}_s${S}_$(echo $ARM | tr -d ' -')
  done
done
```

Three seeds per arm, because one run cannot separate a real effect from run-to-run noise at these
margins. The classical arm (empty `ARM`) is shared, so a second site costs only two more runs
per seed.

### Rate–distortion curve

```bash
for L in 5e-4 7.5e-4 1e-3 1.5e-3 2.5e-3; do
    python train_dyna.py --gpu_ids 0 --num_workers 4 --lambda_reward $L --seed 0 --val_size 2000
    python make_figures.py --gpu_ids 0 --lambda_reward $L --seed 0 --val_size 2000 \
        --results_dir ./figures/lambda_$L
done
```

~17 h for five points on one arm.

---

## Converting an external checkpoint

For a single-file `{SE,CE,G,P}` checkpoint, such as one produced by the Kaggle notebook:

```bash
python convert_kaggle_weights.py --src Weight/final.pth --lambda_reward 1.5e-3 --force
```

Pass the same `--C_channel` / `--lambda_L2` / `--lambda_reward` / `--select` you will later use, as
they decide where it lands. Each state dict is validated with `strict=True` before anything is
written.

> A converted checkpoint has no `--seed`, so evaluate it **without** one.
> Folders from before the naming fix are `C16_L2_1_re_<…>`; rename to `C16_L2_1.0_re_<…>`.

## Kaggle

`Kaggle_Train_and_Figures.ipynb` clones this repo and runs the same scripts, with a `MODE` switch
for training here or loading an attached checkpoint. Needs **GPU** and **Internet** on, and never
install the pinned `torch` from `requirements.txt` there.

---

See [CHANGES.md](CHANGES.md) for what differs from the original code, and
[Quantum_JSCC_Plan.md](Quantum_JSCC_Plan.md) for why the experiments are shaped this way.
