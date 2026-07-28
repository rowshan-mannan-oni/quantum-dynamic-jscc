# Hybrid Quantum–Classical Dynamic JSCC — Research Plan

**Base work:** Yang & Kim, *Deep JSCC for Wireless Image Transmission with Adaptive Rate Control*
(ICASSP 2022, arXiv:2110.04456), reproduced in this repository.

**Extension:** insert a trainable variational quantum circuit (VQC) into the JSCC pipeline and
measure what happens, across several candidate insertion sites, against a matched classical
baseline.

> Scope note: this document previously covered only the channel encoder. It now covers the whole
> design space, because *which* site a VQC can usefully occupy became the more interesting
> question — see §2.

---

## 0. Status

| Component | State |
|---|---|
| Classical baseline, 400 epochs (α=1.5e-3, `hard`) | **done** — 24.35 dB @ 0 dB SNR to 30.93 dB @ 20 dB, CPP 0.49 → 0.25, SSIM 0.932 @ 10 dB |
| Figures + CSVs (`make_figures.py`), fixed-rate override | **done** |
| Resumable training (optimizer, epoch, Gumbel temperature, stage logic) | **done** |
| Kaggle notebooks (train and/or figures, driven from this repo) | **done** |
| TorchQuantum install validated: GPU forward **and backward**, per-site benchmark | **done** — see `quantum/README.md` |
| `quantum/` scaffold: circuit primitives, site registry, `--quantum_site` | **done** |
| Site modules in `quantum/modules.py` | **pending** — next step |
| Integration hooks in `models/` | **pending** |
| Comparison runs | **pending** |

---

## 1. Framing and honest scope

This is exploratory research, not a drop-in improvement. **A PSNR gain over the classical
baseline is unlikely and is not the claim.** Variational circuits have no established advantage
on classical data at this scale, and the optimisation is strictly harder (barren plateaus, slower
convergence). Building the thesis on "quantum wins" would most likely fail and would invite
justified reviewer scepticism.

The defensible contribution is a **systematic map of where a VQC can be placed inside a deep JSCC
pipeline, and what each placement costs and changes.** That question has not been answered, and
it stands whether or not any site improves PSNR. Parity at a comparable or smaller parameter
count is a real result.

Success criteria, in priority order:

1. A hybrid model trains stably end to end and reconstructs images within a few dB of classical.
2. Adaptive rate control still emerges — rate falls as SNR rises, as in the base paper.
3. A fair, toggleable A/B/C against classical and classical-matched controls, reporting
   PSNR/CPP curves, parameter counts, wall-clock and circuit gradient statistics.
4. A defensible account of *why* each site behaves as it does.
5. (Stretch) A site where the quantum module matches or beats classical at some operating point.

---

## 2. The design space

The pipeline, with tensor shapes for the default configuration:

```
image (N,3,32,32)
  └─ netSE  source encoder ──────────────▶ z (N,256,8,8)
  └─ netCE  channel encoder
       res1 ─ mod1(SNR) ─ res2 ─ mod2(SNR) ─ projection Conv3x3(256→16)
                                          ──▶ latent (N,16,8,8) = 8 groups x 128
  └─ netP   policy: pool(z)+SNR → MLP → Gumbel-Softmax → thermometer mask
  └─ power normalise, mask, AWGN
  └─ netG   decoder: mask_conv(16→256) ─ res ─ mod(SNR) ─ upsample ──▶ (N,3,32,32)
```

Candidate sites, with cost measured on an RTX 4070 Laptop at 8 qubits, 2 layers
(`python -m quantum.smoke_test --bench`; classical baseline is ~0.5 min/epoch):

| # | Site | Replaces | Batch seen by circuit | +min/epoch | Claim it supports |
|---|---|---|---|---|---|
| **A** | Policy network | `netP.model_gate` MLP | 128 | +0.3 | Quantum variational policy for adaptive rate control |
| **B** | Channel-encoder projection | `netCE.projection` Conv3×3 | 8192 | +0.5 | **Transmitted symbols are produced by a circuit** |
| **C** | Decoder input | `netG.mask_conv` | 8192 | +0.5 | Quantum receiver; pairs with B for a "quantum link" |
| **D** | SNR modulation | `modulation` MLPs (×4) | 128 | +0.3 | Quantum channel-state-aware modulation |
| **E** | The channel itself | AWGN model | — | low | Semantic communication over a quantum channel |
| **F** | Quanvolution in `netSE` | early conv | very large | high | (well-trodden; not pursued) |

**The measurement that shapes the plan:** at 8 qubits a batch of 8192 costs about what 128 does
(76 ms vs 44 ms per step). Statevector simulation at this width is bound by kernel launches, not
arithmetic, so folding the 64 spatial positions of an 8×8 feature map into the batch dimension is
nearly free. That is what makes site B — the site with the strongest claim — affordable at all:
roughly 6.5 h for 400 epochs against 3.4 h classical, rather than the multi-day run a
position-by-position loop would require.

**Exploration order:** A (cheap end-to-end proof that autograd survives the Gumbel-Softmax path)
→ D → **B (the headline)** → C. F is not planned. E is deferred: it is the most novel option but
it changes the task rather than the architecture, which muddies the comparison; better as
follow-on work.

---

## 3. The constraint every site must respect

A circuit transforms a state, but **getting a wide classical vector in and out is the hard part.**
Encoding costs one angle per qubit and measurement returns one expectation per qubit, so 1024
latent values cannot be routed through an 8-qubit state and recovered. Every quantum module is
therefore a sandwich:

```
Linear(in → q) → tanh·π → VQC → MeasureAll(PauliZ) → Linear(q → out)
```

implemented once as `quantum.layers.HybridVQC`, so all sites share one circuit design. Outputs
are expectation values in [-1, 1]; whatever consumes them must expect that scale.

---

## 4. Reference module — `QuantumProjection` (site B)

```
z (N,256,8,8)
  └─ fold spatial into batch      ──▶ (N*64, 256)      each position is a 256-vector
  └─ [optional concat SNR]        ──▶ (N*64, 257)
  └─ Linear(→ q), tanh·π          ──▶ (N*64, q)        bounded angles
  └─ RY angle encoding, then L × [RY,RZ per wire + ring CNOT]
  └─ MeasureAll(PauliZ)           ──▶ (N*64, q) ∈ [-1,1]
  └─ Linear(q → 16)               ──▶ (N*64, 16)
  └─ unfold                       ──▶ (N,16,8,8) = latent
```

**Parameter budget (q=8, L=2):** pre-linear ≈ 2.1k, circuit angles 32, post-linear 144 → **≈ 2.2k**,
against **≈ 37k** for the classical Conv3×3. The quantum module is *smaller*, which is exactly why
the classical-matched control in §5 is mandatory.

Alternatives if this underperforms: 2×2 quanvolution (more spatial structure, heavier), or a
quantum residual branch alongside the classical projection (lowest risk, weakest claim).

---

## 5. Fair comparison

Three arms through **identical** training and evaluation code:

1. **Classical** — the original module. The reference.
2. **Classical-matched** — e.g. a 1×1 conv at roughly the quantum parameter count. Without this,
   any difference can be dismissed as "you shrank the layer".
3. **Quantum**.

Held identical across arms: dataset and splits, channel model, SNR range, optimiser, batch size,
epoch schedule, data order, and random seeds. Report mean and spread over **≥3 seeds** — a single
run cannot distinguish a real effect from noise at these margins.

Logged per arm: PSNR/CPP vs SNR, per-class rate, parameter count, wall-clock per epoch, circuit
gradient variance.

**Both arms must be shown to have converged.** If one plateaus and the other has not, the
comparison is confounded by under-training regardless of how symmetric the protocol looks. Include
the loss curves.

Absolute numbers may differ from the published paper because of protocol choices; that is fine and
should be stated. The contribution is the *relative* comparison, with the paper's numbers cited as
a reference point.

---

## 6. Code structure — plugin, not fork

Everything quantum lives in `quantum/`; `models/` stays shared:

```
quantum/
  layers.py       VariationalCircuit, HybridVQC        — circuit primitives
  modules.py      per-site drop-in modules             — registered via @register_site
  smoke_test.py   install validation + per-site benchmark
  __init__.py     site registry
```

Selected at run time with `--quantum_site {none,...}`, `--n_qubits`, `--vqc_layers`. Default
`none` leaves the model exactly classical.

**Why not duplicate the tree into a second folder:** the comparison's credibility rests on the two
arms differing *only* in the swapped module. Two copies drift — a fix applied to one and not the
other silently turns "quantum vs classical" into "codebase A vs codebase B", which is the first
thing a reviewer will probe. Three real bugs were fixed here during setup (`SSIM` never computed,
per-class statistics aliased through `[[]]*10`, checkpoint naming); each would have had to be
caught twice.

Each site module must be a drop-in with identical input/output shapes, so nothing downstream
changes. An unrecognised `--quantum_site` raises rather than falling back, so a typo cannot
quietly train a classical model that is later written up as a quantum result.

Separability: `quantum/` is self-contained and can be lifted out whole. Note that "delete the
classical files" is not quite meaningful — the model is a *hybrid*, and still needs the CNN
encoder, decoder, policy and training loop. Going quantum-only means deleting one classical branch
and the flag.

---

## 7. Training strategy

- **Separate parameter group for circuit angles.** They typically want a higher learning rate than
  the classical weights; start around 5e-3 against the paper's 5e-4 and tune.
- **Keep the paper's three stages** (joint 5e-4 → decay 5e-5 → fine-tune 1e-5 with `netSE` and
  `netP` frozen). Note that site A lives in `netP`, which the fine-tune stage freezes — so a
  quantum policy stops training a quarter of the way through unless that is changed deliberately.
  Sites B/C/D live in modules that keep training throughout.
- **Watch the policy.** A poorly-trained encoder can collapse the Gumbel-Softmax decision to
  maximum rate. Track decision entropy early.
- **Watch for barren plateaus.** Log gradient norm and variance of the circuit angles per epoch,
  and the mean |Z| of measurements to catch saturation.
- **Validate small first.** One forward/backward, then a few epochs on a subset, before committing
  to a full run.

---

## 8. Compute budget

Measured on an RTX 4070 Laptop; re-measure on the target GPU rather than scaling these.

| Configuration | min/epoch | 400 epochs |
|---|---|---|
| Classical | ~0.5 | ~3.4 h |
| Quantum, site A or D (q=8) | ~0.8 | ~5.3 h |
| Quantum, site B or C (q=8, L=2) | ~1.0 | ~6.5 h |
| Quantum, site B (q=8, L=4) | ~1.4 | ~9 h |

Memory is not the constraint — a 10-qubit statevector is tiny. Throughput is.

Training will run on a Linux workstation with an RTX 3090. On Kaggle, a full quantum run may
approach the session limit, which is why **resume is now implemented and correct**: optimizer
state, completed epoch and Gumbel temperature are checkpointed, and the learning-rate stage is
derived from the epoch rather than triggered on a boundary. A λ-sweep of multi-hour jobs is
otherwise fragile.

---

## 9. Environment

Full recipe and verified versions: **`quantum/README.md`**. In short — install TorchQuantum from
**GitHub, not PyPI**, and always with `--no-deps`:

```bash
pip install --no-deps git+https://github.com/mit-han-lab/torchquantum.git
pip install -r requirements-quantum.txt
```

The PyPI release (0.1.8) imports `qiskit.providers.aer`, removed in qiskit 1.0, so it cannot run
against a current qiskit at all. GitHub `main` (0.3.0) uses `qiskit_aer`. `--no-deps` is required
on every platform because the declared dependencies pull in `tensorflow` and `pyscf`, neither of
which statevector simulation uses.

Verified: torchquantum 0.3.0, qiskit 2.5.1, qiskit-aer 0.17.2, torch 2.5.1+cu124, RTX 4070 —
circuits run on GPU and gradients reach their parameters.

**Validate before building on it:** `python -m quantum.smoke_test --bench`. A successful import
proves very little; a silent gradient failure is indistinguishable from a model that will not
learn.

---

## 9a. Compatibility

This stack sits between a 2021 codebase, a GPU generation that codebase predates, and a quantum
library that tracks neither. Most of the setup time so far went here, and every item below was hit
in practice rather than anticipated.

### Verified working set

| Package | Version | Note |
|---|---|---|
| torch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 | CUDA wheel index, not PyPI default |
| numpy | 2.2.6 | see the numpy trap below |
| scikit-image | 0.25.2 | must be ≥0.24 under numpy 2 |
| matplotlib | 3.10.9 | |
| torchquantum | 0.3.0 (GitHub `main`) | **not** the PyPI release |
| qiskit / qiskit-aer / qiskit-ibm-runtime | 2.5.1 / 0.17.2 / 0.48.0 | |
| setuptools | 83.0.0 | no pin needed with the GitHub install |

Verified on Windows 11 + RTX 4070 Laptop (sm_89). The Linux + RTX 3090 (sm_86) target should be
easier, not harder: the one Windows-specific failure (`pyscf`) disappears there.

### The traps, and why each bites

**1. The original PyTorch cannot run on your GPU.** Not a preference — 2021 builds have no
compiled kernels for sm_86/sm_89. Any "reproduce the original environment" instinct fails here.

**2. TorchQuantum's PyPI release is unusable.** 0.1.8 (Feb 2024) imports
`qiskit.providers.aer`, a path removed in qiskit 1.0, at module scope. No qiskit version
satisfies both it and a modern install. GitHub `main` (0.3.0) uses `qiskit_aer` and works with
qiskit 2.x. **Install from GitHub.**

**3. `--no-deps` is mandatory on every platform.** TorchQuantum's declared dependencies include
`tensorflow` and `pyscf`, neither used by statevector simulation. `pyscf` has no Windows wheels
and needs a C++ toolchain, so the install fails outright there; on Linux it merely wastes a
gigabyte and puts a second framework's CUDA runtime next to PyTorch's.

**4. numpy 2 is forced on you, and it breaks compiled packages.** Installing the quantum extras
pulls qiskit, which upgrades numpy to 2.x — silently, after `requirements.txt` pinned 1.26.4.
Anything compiled against the numpy 1.x ABI then fails at *import*:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
            Expected 96 from C header, got 88 from PyObject
```

This hit scikit-image 0.22 and would have broken `test_dyna.py`'s SSIM the first time it ran
after the quantum install. `requirements.txt` now requires `numpy>=2` and `scikit-image>=0.25`
so both halves agree, rather than pinning numpy low and having it overridden.

**5. qiskit ≥1.0 cannot coexist with `qiskit-terra`.** Upgrading over an older qiskit raises
`ImportError: Qiskit is installed in an invalid environment`. Uninstall `qiskit`, `qiskit-terra`
and `qiskit-aer` before installing the modern stack; a plain `--upgrade` does not fix it.

**6. `pkg_resources` was removed in setuptools 81.** Only affects the old pinned path; the
GitHub install with a current `qiskit-ibm-runtime` needs no setuptools pin.

**7. Windows dataloader workers.** The scripts execute at module scope with no
`if __name__ == '__main__'` guard, so spawned workers re-import them and the run dies. Hence
`--num_workers` defaulting to 0. **On the Linux box, set it to 2–4** — with CIFAR-10's small
images, data loading is otherwise the bottleneck.

**8. Kaggle: never install the pinned torch.** Its image ships a working CUDA build; replacing it
breaks the GPU. Install only scikit-image, matplotlib and the quantum extras there.

### Both arms share one environment

The classical and quantum arms run in the same environment, which is what makes the comparison
practical — but it also means a quantum-side dependency change can break the classical arm, as
the numpy episode shows. After touching quantum dependencies, re-check both:

```bash
python -m quantum.smoke_test          # circuit runs, gradients flow
python test_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --SNR 10 --num_test 300
```

Last verified together: PSNR 28.98, SSIM 0.933, 5.13 groups at 10 dB, with the smoke test passing.

### Checkpoint compatibility

- **Notebook checkpoints** pack all four sub-networks into one file. `convert_kaggle_weights.py`
  splits them into the per-network layout, validating each with `strict=True`.
- **Pre-resume checkpoints** have no `*_training_state.pth`. `--continue_train` still loads the
  weights and warns that the optimizer and temperature restart cold, rather than failing.
- **Folder names** encode `C_channel`, `lambda_L2`, `lambda_reward` and `select`. Checkpoints
  written before the `lambda_L2` fix are named `C16_L2_1_re_<...>` and must be renamed to
  `C16_L2_1.0_re_<...>` to be found.
- **`torch.load` will default to `weights_only=True`** in a future release. Our files hold
  tensors and plain scalars, so this should be safe, but it is worth re-testing on the next major
  torch upgrade.

---

## 10. Evaluation

`make_figures.py` already produces, per configuration:

- Average rate (CPP) and PSNR vs SNR
- Attainable PSNR at each fixed rate, with adaptive points overlaid
- Per-class rate and PSNR
- Sample reconstructions, plus the CSVs behind every figure

Quantum-specific additions still to build: the classical / classical-matched / quantum overlay,
a parameter-count and wall-clock table, and circuit gradient-variance curves.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| A quantum dependency silently breaks the classical arm | Shared environment, so re-run both checks after any dependency change — §9a |
| Barren plateaus | 2 layers, near-identity initialisation, bounded encoding, log gradient variance |
| Quantum module tanks PSNR | Start as a residual branch, or warm-start classical then swap; separate learning rate |
| Policy collapses to maximum rate | Track decision entropy; extend the joint stage; tune α |
| Under-training confounds the comparison | Show converged loss curves for both arms |
| "You just shrank the layer" | Classical-matched control; report all parameter counts |
| Single-seed noise mistaken for effect | ≥3 seeds, report spread |
| Session limits on long runs | Resume is implemented; validate at reduced epochs first |
| No PSNR gain anywhere | Expected. The contribution is the design-space map, §1 |

---

## 12. Open decisions

1. **Which site to implement second**, after A proves the plumbing — D (cheap, in-path) or straight
   to B (the headline).
2. **Qubit count**: 8 recommended; 6 if throughput bites. Cost scales as 2^q.
3. **Feed SNR into the quantum pre-linear?** Recommended yes — cheap, keeps the module
   channel-aware.
4. **Site A and the fine-tune freeze.** `netP` is frozen for the last 100 epochs; a quantum policy
   would stop learning there. Keep the schedule (and accept it) or exempt the site.
5. **Site A and Fig. 5.** The fixed-rate figure overrides the policy via `forced_active`, so with a
   quantum policy that figure no longer exercises the quantum module. Decide what it means for
   that arm.
6. **How many λ values** for the tradeoff curve, given each is a separate multi-hour run per arm.

---

## 13. Milestones

1. ~~**Environment** — install TorchQuantum, verify a batched circuit forward *and backward* on
   GPU.~~ **done**
2. ~~**Scaffold** — `quantum/` package, circuit primitives, site registry, `--quantum_site`.~~ **done**
3. **Module** — implement the first site module; unit-test shapes and confirm gradients reach the
   circuit angles through the full model.
4. **Integrate** — add the factory hook at that site; run a few epochs on a subset end to end.
5. **Short run** — ~30–60 epochs, quantum vs classical, check PSNR/CPP trends and stability logs.
6. **Full runs** — full schedule for the chosen configurations and seeds; all figures plus the
   quantum-specific reporting.
7. **Analysis** — overlays, parameter and wall-clock table, gradient-variance discussion; decide
   whether to escalate qubits or depth, or to move to the next site.

---

*Infrastructure and environment are in place; §13.3 is the next step.*
