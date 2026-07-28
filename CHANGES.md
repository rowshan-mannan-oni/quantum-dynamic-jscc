# Changes to the original code

A running, high-level record of everything changed since the fork of
[mingyuyng/Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC). Newest first.
Detail lives in the commit messages; this is the map.

**Legend:** 🐞 bug fix · ✨ new capability · ⚙️ infrastructure · 📄 docs · ⚠️ changes behaviour

---

## Quantum work

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
