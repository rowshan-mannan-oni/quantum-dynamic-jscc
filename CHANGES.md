# Changes to the original code

A running, high-level record of everything changed since the fork of
[mingyuyng/Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC). Newest first.
Detail lives in the commit messages; this is the map.

**Legend:** 🐞 bug fix · ✨ new capability · ⚙️ infrastructure · 📄 docs · ⚠️ changes behaviour

---

## Quantum work

### 2026-07-29 — `Dynamic_JSCC_Kaggle.ipynb` rewritten to train an arm from the repository

| | |
|---|---|
| ⚠️ | **The notebook no longer carries its own copy of the model.** It was a standalone reimplementation that had drifted from the repository — no quantum site, no seeding, no held-out validation, and its own `DynaJSCC` class. It now clones the repo and calls `train_dyna.py`, `make_figures.py` and `compare_arms.py`, so it always reflects the latest commit. |
| ✨ | One `ARM` switch selects `classical` / `matched` / `quantum`; run the notebook three times to build the comparison. Installs `torchquantum` (from GitHub, `--no-deps`) and runs `quantum.smoke_test` before committing hours to a circuit that might not be receiving gradients. |
| ✨ | The run folder is derived by importing `run_name` from `options.base_options` rather than rebuilding the string, which is what the previous notebook did — it would have silently written to the classical folder while training a quantum arm. |
| ✨ | Resume across Kaggle's session limit: attach a previous run's `Checkpoints/<run>/` as an input and it restarts from the last rolling checkpoint. Also parses `G_reward` after training and prints a **loud warning if the policy has collapsed** to a constant rate. |

### 2026-07-29 — Dead `Normalize` class removed

| | |
|---|---|
| 🐞 | **`networks.Normalize` called `np.sqrt` in a file that never imports numpy**, so any use of it raised `NameError`. Nothing instantiated it — inherited unused from the fork — which is why it never surfaced. Deleted rather than fixed: the model already power-normalises inline in `DynaAWGN_model.forward`, per channel over the last two dims, where the dead class averaged over all channels at once. |

### 2026-07-29 — The angle encoder saturated, so both site modules now normalise their input

| | |
|---|---|
| 🐞 | **`HybridVQC` and `MatchedBottleneck` fed their input straight into `tanh(Linear(x)) * pi`.** At the policy site that input is 256 pooled activations averaging 0.57 alongside one SNR feature spanning 0–20, so the SNR term dominated the scale: in the trained `qpolicy-q8l2_s0` checkpoint 56% of encoder units were saturated at 0 dB and **82% at 20 dB**, the circuit's output varied by 5e-3 across 512 different images, and the winning logit led by ~20 against Gumbel noise of std ≈1.3. The policy could not respond to anything. |
| ⚙️ | Both modules now start with `BatchNorm1d(in_features, affine=False)` — per-feature, so the one outsized feature is rescaled instead of dominating, and parameter-free, so the site's budgets (quantum 2,141 · matched 2,181) are unchanged. Applied to **both** arms: fixing only the quantum one would have stopped the control being a control. |
| ⚠️ | **`qpolicy` and `mpolicy` checkpoints from before this do not load** — `netP` gains `norm.running_mean` / `norm.running_var`, so loading fails with *Missing key(s) in state_dict*. `netSE`, `netCE` and `netG` are unaffected. Retrain the affected arms. |

### 2026-07-29 — `compare_arms.py`, and the first quantum-vs-classical comparison

| | |
|---|---|
| ✨ | **`compare_arms.py`** overlays the sweeps of two or more arms from the csv files `make_figures.py` leaves behind, writing `comparison_rate_psnr.png` and `comparison_data.csv`. Rate and PSNR get a panel each rather than a twin axis. It reports any arm whose policy has collapsed to a constant rate, and pulls in the first arm's fixed-rate curve at that rate so the PSNR panel is like-for-like. |
| ⚠️ | **The `qpolicy-q8l2_s0` policy has collapsed:** it selects exactly 7 groups (CPP 0.438) for every image at every SNR, so it is a fixed-rate codec, not an adaptive one. The classical baseline moves 0.492 → 0.254 over the same sweep. |

### 2026-07-29 — `opt.txt` records the command that trained the run

| | |
|---|---|
| ✨ | **`opt.txt` now opens with the start time and the exact command line**, quoted so it pastes back (`subprocess.list2cmdline` on Windows, `shlex.quote` elsewhere). Since evaluation takes no checkpoint path and rebuilds the folder name from the flags, this is where to read those flags back from. |
| 📄 | `RUNS.md` rule 1 points at `opt.txt` for recovering the flags, and warns that training-only ones must be dropped for `make_figures.py`. |

### 2026-07-29 — Full argument reference, and a silent-control bug it exposed

| | |
|---|---|
| 📄 | `RUNS.md` rewritten: every training and inference argument documented, and a section per site with the commands for each arm. Sites not yet built are marked in their headings. |
| 🐞 | **`--matched_site <unbuilt site>` trained a plain classical model** into a folder named as though it were the control. Only `--quantum_site` was validated. Both are now checked against the sites that actually have a hook, and `build_swappable` asserts the two lists agree. |
| 📄 | Documented that `--ndf`, `--max_dataset_size` and `--phase` are inherited from the CycleGAN codebase and never read. |

### 2026-07-29 — Best-checkpoint tracking

| | |
|---|---|
| ✨ | **A run now keeps two sets of weights:** `latest_*` overwritten every `--save_epoch_freq` epochs, and `best_*` kept whenever the score improves. Evaluate either with `--epoch best` / `--epoch latest`. |
| ✨ | `--val_size N` holds N images out of training to score on. Left at 0 it scores the epoch's training loss instead. The score is the training objective, not PSNR alone, so a model is not called better for reconstructing well while over-transmitting. |
| ✨ | `--snapshot_freq` — numbered snapshots are now opt-in and separate from the rolling save. |
| ⚠️ | `--save_epoch_freq` default **40 → 5**, and it no longer writes numbered snapshots (that is `--snapshot_freq`, default off). `--save_latest_freq 0` now disables the mid-epoch save. |
| 🐞 | `latest` is now also written at the final epoch, so a run whose length is not a multiple of `--save_epoch_freq` no longer ends with a stale rolling checkpoint. |
| 🐞 | Resuming reads the best score from the best checkpoint rather than the rolling one, which could otherwise be older and let a worse epoch overwrite a better `best`. |

### 2026-07-29 — Run cookbook, and an SSIM crash it exposed

| | |
|---|---|
| 📄 | `RUNS.md` — the command for each result: training each arm, evaluation, figures, sweeps, resuming, Kaggle. |
| 🐞 | **`test_dyna.py` crashed at the default `--num_test_channel 5`.** The SSIM loop added earlier indexed the original image once per channel realisation, but only one original exists — PSNR broadcasts, the loop did not. Every earlier test happened to pass `--num_test_channel 1`, which hid it. |

### 2026-07-29 — Swappable sites and the first quantum module `07cf524`

| | |
|---|---|
| ✨ | **Choosing an arm is one flag.** `--quantum_site policy` swaps in a circuit, `--matched_site policy` swaps in a same-size classical control, neither flag leaves the model as it was. |
| ✨ | **First site implemented:** quantum policy network (`quantum/modules.py`). 2,141 params against the classical 21,253. Costs 48 s/epoch vs 31 s. |
| ✨ | `MatchedBottleneck` in `models/networks.py` — the classical control arm, 2,181 params, within 2% of the circuit so a difference between them is attributable to the circuit and not to the smaller layer. |
| ⚠️🐞 | **Checkpoint folder names now encode the arm.** They did not, so a quantum run resolved to the same folder as the classical baseline and would have overwritten it silently. Classical names are unchanged, so existing checkpoints still load. |
| 🐞 | The folder name is now built by one function instead of four copies of the same expression across `train_dyna`, `test_dyna`, `make_figures` and `convert_kaggle_weights`. |
| ✨ | `--seed` seeds torch, numpy and random. Unset keeps the old unseeded behaviour. Set it for anything being compared. |
| ⚙️ | Unknown site names, and setting both site flags at once, now fail before training starts instead of quietly training something else. |
| ⚙️ | `opt.txt` written beside the checkpoints as a provenance record of the run. |

### 2026-07-29 — Source encoder site re-costed `1c2b64d`

📄 The plan had dismissed a quantum source encoder as too expensive. That is only
true for overlapping quanvolution (+19 min/epoch). 4×4 non-overlapping patches land
on exactly 8×8 — `netSE`'s own output shape — for **+0.6 min/epoch**. Site added to
the exploration order.

### 2026-07-28 — Environment compatibility `fe17efd` `496fe45` `29058b4`

| | |
|---|---|
| ⚙️ | `quantum/` package: circuit primitives (`layers.py`), site registry, and `smoke_test.py` that checks gradients actually reach circuit parameters, not just that imports work. |
| ⚠️ | **TorchQuantum must come from GitHub, not PyPI, always with `--no-deps`.** The PyPI release imports a qiskit path removed in 1.0 and cannot run against a current qiskit. |
| 🐞 | **numpy 2 break.** Installing the quantum extras pulls qiskit, which upgrades numpy to 2.x and breaks anything built against the 1.x ABI — scikit-image 0.22 included, which would have crashed `test_dyna.py`'s SSIM. Requirements now ask for `numpy>=2` and `scikit-image>=0.25` so both halves agree. |
| 📄 | `Quantum_JSCC_Plan.md` (renamed from `..._Encoder_Plan.md`): the research plan, measured per-site costs, and a compatibility section recording every trap hit so far. |

---

## Reproduction infrastructure

### 2026-07-28 — Resumable training `0b0d05a`

| | |
|---|---|
| ⚠️🐞 | **Resume was silently wrong.** Stage transitions tested `epoch == boundary + 1`, which is never true when resuming past a boundary, so a restarted run continued on the wrong learning rate and never froze the networks it should have. The stage is now derived from the epoch. |
| 🐞 | Optimizer state, completed epoch and Gumbel temperature are now checkpointed. Previously Adam restarted cold and the temperature reset to its initial value. |
| ✨ | `--continue_train` picks up the right epoch by itself. Checkpoints predating this still load, with a warning. |

### 2026-07-28 — Checkpoint naming `c547102`

⚠️🐞 `argparse` applies `type=` to command-line values but not to a non-string default,
so `--lambda_L2` gave `L2_1` when omitted and `L2_1.0` when passed — two folder names
for the same run. Defaults are floats now. **Folders named `C16_L2_1_re_<…>` must be
renamed to `C16_L2_1.0_re_<…>`.**

### 2026-07-28 — Kaggle notebooks `316f777` `fa78cee`

| | |
|---|---|
| ✨ | `Kaggle_Train_and_Figures.ipynb` — clone the repo, train *or* load weights, then plot. One `MODE` switch. |
| ✨ | `Kaggle_Figures.ipynb` — figures only, from an existing checkpoint. |
| | Both clone from GitHub rather than embedding a copy of the code, so they track the latest commit. |

### 2026-07-28 — Figures and weight conversion `5326d0f` `316f777`

| | |
|---|---|
| ✨ | `make_figures.py` — the paper's figures plus the CSVs behind them. The repo had no plotting code at all. |
| ✨ | `forced_active` on the model, to pin the rate instead of letting the policy choose. The fixed-rate figure is impossible without it. |
| ✨ | `convert_kaggle_weights.py` — converts a single-file `{SE,CE,G,P}` checkpoint into the per-network layout, validating each with `strict=True`. |
| 🐞 | **`Mean SSIM` always printed `nan`** — `SSIM_list` was never populated. Now computed. |
| 🐞 | **Per-class statistics were all identical** — `[[]]*10` aliases one list ten times, so every class printed the global mean. |

### 2026-07-20 — Environment `3e15d0e`

| | |
|---|---|
| ⚠️ | Modern PyTorch is **required**, not preferred: the original 2021 build has no kernels for Ampere or Ada (RTX 3090, 4070) and will not run at all. |
| 🐞 | **`num_workers` on Windows.** The scripts run at module scope with no `__main__` guard, so spawned dataloader workers re-import them and the run dies. Default is now 0; **set 2–4 on Linux**, where data loading is otherwise the bottleneck. |
| ⚙️ | `requirements.txt`, `LICENSE` (CC BY-NC-SA 4.0, inherited), `.gitignore`, rewritten `README`. |

---

## Things to watch

- **Fine-tuning freezes `netP` and `netSE`** for the last 100 of 400 epochs, so a
  quantum module at either site stops learning a quarter of the way through.
  Undecided.
- **The fixed-rate figure bypasses the policy**, so it does not exercise a quantum
  policy at all. Needs a decision on what that figure means for that arm.
- **Resume is not a substitute for one clean run.** It is correct now, but numbered
  checkpoints carry no optimizer state; only `latest` can be resumed from.
