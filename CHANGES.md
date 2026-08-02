# Changes to the original code

A running, high-level record of everything changed since the fork of
[mingyuyng/Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC). Newest first.
Detail lives in the commit messages; this is the map.

**Legend:** 🐞 bug fix · ✨ new capability · ⚙️ infrastructure · 📄 docs · ⚠️ changes behaviour

---

## Quantum work

### 2026-08-02 — `metrics.csv`: what the circuit was doing while it trained

| | |
|---|---|
| ✨ | **Training now writes `metrics.csv` beside the checkpoints, one row per epoch:** rate entropy, angle-encoder saturation, mean `\|⟨Z⟩\|`, and the gradient norm *and variance* on the circuit angles. `loss_log.txt` says *that* a run went wrong; none of these survive into the checkpoint, so after the run the information simply does not exist anywhere. Plan §5, §7 and §11 all asked for it — §11 names gradient variance as the barren-plateau mitigation — and none of it was implemented. |
| ✨ | **Columns are per network**, so `--quantum_site projection,decoder` reports `ce_*` and `g_*` separately and you can see which end moved. Sites are found by structure (any module carrying a `.pre` linear), so a new site is covered the moment it is registered. |
| ✨ | Training **warns at the end of any epoch** where rate entropy collapses to zero or gradient variance vanishes, so a doomed run does not depend on someone reading the file. |
| ⚙️ | Nothing touches the graph, the loss or any weight — hooks read forward activations, gradients are read after the optimizer step, and the accumulators are switched off around validation so a fixed-SNR eval pass cannot contaminate an epoch's training statistics. Measured overhead: none (classical still 30 s/epoch). |

### 2026-08-02 — Site C: the decoder input, and the quantum link

| | |
|---|---|
| ✨ | **`--quantum_site decoder`** replaces `netG.mask_conv`, `Conv3×3(16→256)` — the first thing the receiver does with what came off the channel. 36,864 params → **2,472 quantum · 2,512 matched**. Note the sandwich is inverted relative to site B: nearly all the budget sits on the output side. |
| ✨ | **`--quantum_site projection,decoder` makes both ends of the link circuits**, which is the strongest claim this pipeline supports. Measured 114 s/epoch against 65 s for the decoder alone and 30 s classical — ~12.7 h for 400 epochs, so it needs more than one Kaggle session. |
| 📄 | **C should be read against B, not alone.** B constricts the transmitted latent to rank 8 of 16; C then expands 16→256 through another 8-measurement circuit. Those do *not* compound — composing two rank-8 maps is still rank 8 — so B has already discarded exactly the dimensions C would, and **C's marginal cost on top of B should be smaller than its cost alone**. What C does add is a second receptive-field loss on the receive side. |
| ⚙️ | The three site factories were collapsed onto one `sandwich()` constructor now that a third identical body existed. Each site keeps its own registration and docstring — the reasoning, shapes and costs are the per-site content — but the circuit is built in one place, which is what plan §3 requires for a difference between placements to be attributable to placement. |

### 2026-08-02 — Site B: the channel-encoder projection

| | |
|---|---|
| ✨ | **`--quantum_site projection` / `--matched_site projection`** replace `netCE.projection`, the `Conv3×3(256→16)` whose output *is* the transmitted latent. This is the site with the strongest claim: every other one shapes or selects what a classical layer transmits, this one produces the symbols. 36,880 params → **2,232 quantum · 2,272 matched**. Measured on an RTX 4070 Laptop at 1.1 min/epoch against 0.6 matched and 0.5 classical, so ~7.3 h for the full 400. |
| ✨ | **`PerPosition`** folds the 8×8 grid into the batch, so all 64 positions run as one circuit evaluation at batch 8192 rather than a 64-step loop — nearly free, because 8-qubit statevector simulation is bound by kernel launches, not arithmetic. It is applied by `build_swappable` **outside** both arms, so the circuit and its control share one implementation of the fold instead of each carrying a copy that could drift apart. |
| ⚠️ | **Both swapped arms are 1×1 where the classical reference is 3×3.** A circuit takes one vector per shot, so the swap also gives up the convolution's neighbourhood. It applies equally to quantum and matched, so *that* comparison stays clean — but part of any gap against the unmodified reference is the lost receptive field rather than the circuit, and it has to be reported next to the parameter count. |
| ⚠️ | **No SNR feature is fed to this site**, unlike `policy`. The `Conv3×3` it replaces never received one — channel state arrives through `mod1`/`mod2` upstream — so feeding it here would hand the arm an input the original did not have. This departs from plan open-decision 3, deliberately. |
| 📄 | Unlike site A, the projection lives in `netCE`, which the fine-tune stage does **not** freeze, so a circuit here trains for all 400 epochs instead of stopping at 300. |

### 2026-08-02 — `quantum.smoke_test` now checks the sites, not just the install

| | |
|---|---|
| ✨ | **Every registered site is built through the same `build_swappable` hook training uses**, for all three arms, at the real batch and shapes: output shape unchanged, gradients reaching the circuit angles, and the matched control still within 10% of the arm it controls for. Previously the test exercised `HybridVQC` in isolation, which would not have caught a mis-wired hook. A site that registers itself but has no entry in the check fails the run rather than going quietly untested. |
| ✨ | **`PerPosition` is pinned against `Conv2d(1×1)`,** which is the same map by definition. Getting its permute and reshape the wrong way round transposes positions against channels *without changing any shape*, so nothing would raise — the model would simply train on scrambled spatial structure and report a worse number, which at this site would read as "the circuit underperforms". |

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
