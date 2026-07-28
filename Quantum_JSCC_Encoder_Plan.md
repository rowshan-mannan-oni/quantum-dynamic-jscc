# Quantum-Enhanced Channel Encoder for Dynamic JSCC — Implementation Plan

**Base work:** Yang & Kim, *Deep JSCC with Adaptive Rate Control* (ICASSP 2022), reproduced in `Dynamic_JSCC_Kaggle.ipynb`.
**Goal of this extension:** insert a trainable variational quantum circuit (VQC) into the *channel encoder* so that the transmitted symbols pass through a quantum layer — turning "quantum JSCC" into a central architectural claim rather than an ablation.

---

## 1. Framing and honest scope

This is an exploratory research contribution, not a safe drop-in improvement. The claim we can credibly make is **architectural**: an end-to-end differentiable hybrid classical–quantum JSCC where the channel symbols are produced by a VQC, trained jointly with the rest of the pipeline under the same Gumbel-Softmax rate-control policy. A PSNR *gain* over the classical baseline is a hoped-for outcome, not the headline; parity at comparable scale, plus a clean analysis of behavior vs SNR and rate, is already a defensible result.

Success criteria, in priority order:
1. The hybrid model trains stably end-to-end and reconstructs images (PSNR within a few dB of classical).
2. Adaptive rate control still emerges (rate drops as SNR rises, as in the base paper).
3. A fair, toggleable A/B against the classical projection, reporting PSNR/CPP curves, parameter counts, and wall-clock.
4. (Stretch) The quantum encoder matches or beats the classical one at one or more operating points.

---

## 2. Where the quantum block goes

Current channel encoder (`ChannelEncoder`) path, input `z` = (N, 256, 8, 8) from the source encoder:

```
z ──▶ res1 ──▶ mod1(SNR) ──▶ res2 ──▶ mod2(SNR) ──▶ projection Conv3x3(256→16) ──▶ latent (N,16,8,8)
```

The `latent` (16×8×8 = 1024 values = 8 groups × 128) **is** the transmitted signal (after per-channel power norm + policy masking + AWGN). We insert the quantum layer **as a replacement for the final `projection`**, so the symbols are literally the output of a VQC. SNR conditioning is preserved because `mod1/mod2` already inject SNR upstream (and we optionally feed SNR into the quantum pre-linear too).

---

## 3. The core constraint (why we can't "just quantize the latent")

A quantum circuit transforms a state, but **reading a high-dimensional classical vector back out is the hard part** — measurement yields only a few expectation values per circuit. You cannot route 1024 numbers through a quantum state and recover 1024 numbers. Therefore the quantum layer must be a **local channel-mixing / quanvolution** operation wrapped in classical pre/post projections, applied in parallel across spatial positions.

---

## 4. Proposed architecture — `QuantumProjection` (recommended default)

A per-location channel-mixing VQC (a "1×1 quanvolution"). Replaces `projection`.

**Data flow** (n_qubits = q, e.g. 8):

```
latent_in  (N, 256, 8, 8)
  └─ reshape spatial→batch ─▶ (N*64, 256)          # each of 64 locations is a 256-vector
  └─ [optional concat SNR]  ─▶ (N*64, 256(+1))
  └─ Linear(256 → q)        ─▶ (N*64, q)           # learned classical reduction
  └─ angle map: a = π·tanh(·) or π·(·)             # keep angles bounded
  └─ GeneralEncoder(RY on q wires)                 # |ψ⟩ from a
  └─ Trainable VQC: L layers × [RY,RZ per wire + ring CNOT]
  └─ MeasureAll(PauliZ)     ─▶ (N*64, q) ∈ [-1,1]  # exact expectations via statevector
  └─ Linear(q → 16)         ─▶ (N*64, 16)          # back to C_channel symbols
  └─ reshape batch→spatial  ─▶ (N, 16, 8, 8) = latent_out
```

**The efficiency trick that makes this trainable:** fold the 8×8 = 64 spatial positions into the batch dimension so *all* locations for the whole minibatch run as **one batched statevector** (bsz = 128×64 = 8192, each a 2^q-dim state). With q=8 that's 8192×256 complex ≈ 16 MB — one circuit execution per forward, not 64. TorchQuantum runs this batched on GPU and backprops through the statevector with **analytic autograd gradients** (no parameter-shift overhead).

**Circuit design (barren-plateau-aware):**
- Shallow: L = 2 layers to start (do *not* stack depth "to be safe").
- Rotations RY+RZ per wire; entangle with a ring of CNOTs.
- Small-angle initialization for VQC params.
- Bounded input angles (tanh·π) so encoding doesn't saturate.

**Parameter budget (q=8, L=2):** pre-linear 256×8 (+SNR) ≈ 2k, VQC 2·2·8 = 32 angles, post-linear 8×16 = 128 → ~2.2k. The classical `projection` (Conv3×3 256→16) is ~37k params, so the quantum version is *smaller*; for a fair A/B we report both counts and optionally add a matched-size classical control (a 1×1 conv 256→16 ≈ 4k).

### Alternatives (open decision — see §12)
- **2×2 quanvolution:** slide a quantum filter over 2×2 patches (encodes local spatial structure, closer to "quantum conv" literature). More expressive, heavier compute, more positions to batch.
- **Quantum residual refinement:** keep the classical projection, add a VQC branch as a residual on the 16-channel latent. Lowest risk to PSNR, weakest "quantum symbol" claim.

---

## 5. Integration and toggling

- Add `QuantumProjection` as an `nn.Module` with the **exact I/O of the classical projection** ((N,256,8,8) + SNR → (N,16,8,8)).
- In `ChannelEncoder.__init__`, choose `self.projection = QuantumProjection(...) if cfg.USE_QUANTUM_ENCODER else nn.Conv2d(...)`.
- Nothing else in the pipeline changes (power norm, masking, AWGN, decoder, policy, losses, figures all reused).
- Config flags: `USE_QUANTUM_ENCODER`, `N_QUBITS`, `VQC_LAYERS`, `Q_FEED_SNR`, `QUANTUM_ARCH ∈ {conv1x1, quanv2x2, residual}`.

---

## 6. Fair baseline comparison

Run three configurations through the **same** training/eval code and the same Fig-4/5/6 machinery:
1. **Classical** (original Conv3×3 projection) — the reference.
2. **Classical-matched** (1×1 conv, ≈ quantum param count) — controls for "did we just shrink the layer?".
3. **Quantum** (`QuantumProjection`).

Report per config: PSNR-vs-SNR, average CPP-vs-SNR, per-class rates, parameter count, and wall-clock/epoch. This gives a crisp "quantum vs classical symbol generator at matched scale" story.

---

## 7. Training strategy

- **Two optimizer param groups:** quantum angles often want a different (frequently higher) learning rate than the classical weights. Start quantum lr ≈ 5e-3 while classical stays at the paper's 5e-4; tune.
- **Schedule:** keep the paper's 3 stages (joint 5e-4 → decay 5e-5 → fine-tune 1e-5, freeze `E_s`+`P`). The VQC lives in `E_c`, so it keeps training through fine-tune. If the quantum layer converges slowly, extend stages 1–2 for the quantum variant.
- **Gumbel-Softmax:** unchanged temperature anneal; watch decision entropy early — a poorly-trained encoder can collapse the policy to max rate.
- **Stability watches:** log gradient norm/variance of the VQC angles per epoch (barren-plateau early warning); log the measurement statistics (mean |Z|) to catch saturation.
- **Validate small first:** confirm one forward/backward and a few epochs on a data subset before committing to the full run.

---

## 8. Compute budget (Kaggle)

- q=8, one batched circuit/forward: expect ~1.5–3× per-epoch slowdown vs classical. Full 400 epochs may be tight within Kaggle's 12 h GPU limit — plan to (a) validate at reduced epochs, then (b) run full-length with checkpoint/resume, or (c) reduce to q=6.
- Memory is not the bottleneck (statevector is tiny at q≤10); throughput is. Ensure TorchQuantum uses the CUDA device.

---

## 9. Environment / dependencies

- **TorchQuantum** latest PyPI is **0.1.8 (Feb 2024)**, pure-Python (`py3-none-any`), MIT HAN Lab. It does **not** actively track the newest PyTorch, so treat install as a real step:
  - Try `pip install torchquantum`; if imports fail against Kaggle's torch (2.x), install from GitHub source (`pip install git+https://github.com/mit-han-lab/torchquantum.git`) which is more current than the PyPI pin.
  - Watch for **qiskit / numpy version conflicts** pulled by torchquantum; pin numpy to Kaggle's if needed.
  - Verify with a 2-qubit smoke circuit + a backward pass on GPU **before** wiring it into the model.
- Everything else (torch, torchvision, matplotlib) is already in the base notebook / Kaggle image.

---

## 10. Evaluation plan

Reuse the existing evaluation and figures for each config:
- **Fig. 4** rate & PSNR vs SNR, **Fig. 6** per-class rates — as already implemented.
- **Fig. 5-style** attainable-PSNR vs fixed rate.
- **New quantum-specific reporting:** param counts, epoch wall-clock, VQC gradient-variance curve, and the classical-vs-classical-matched-vs-quantum overlay.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Barren plateaus (vanishing VQC gradients) | Shallow circuit (L=2), small-angle init, bounded encoding, monitor grad variance |
| Quantum layer tanks PSNR early | Start as residual, or warm-start classical then swap; separate/ higher quantum lr |
| Policy collapses to max rate | Watch decision entropy; extend joint stage; tune α |
| TorchQuantum ↔ torch incompatibility | Install from GitHub source; isolated smoke test first; pin numpy/qiskit |
| Training too slow for 400 epochs | Reduced-epoch validation → checkpoint/resume full run, or q=6 |
| "Unfair" comparison critique | Include classical-matched (1×1) control; report all param counts honestly |

---

## 12. Open decisions to lock before coding

1. **Architecture variant:** `conv1x1` channel-mixing VQC (recommended) vs `quanv2x2` vs `residual`.
2. **Qubit count:** 8 (recommended), 6, or 4 — trades expressivity vs speed (cost ∝ 2^q).
3. **VQC depth:** start at L=2 (recommended).
4. **Feed SNR into the quantum pre-linear?** (recommended yes — cheap, keeps the layer SNR-aware).
5. **Which configs to actually train given the time budget** (classical + quantum minimum; add classical-matched if time allows).

---

## 13. Phased milestones

1. **Env** — install TorchQuantum on Kaggle, run an isolated q-qubit smoke circuit with a backward pass on GPU.
2. **Module** — implement `QuantumProjection` (conv1x1), unit-test shapes: (N,256,8,8)+SNR → (N,16,8,8); confirm gradients flow to VQC angles.
3. **Integrate** — add the toggle to `ChannelEncoder`; run 2–3 epochs on a data subset to confirm end-to-end training.
4. **Short run** — ~30–60 epochs, quantum vs classical, sanity-check PSNR/CPP trends and stability logs.
5. **Full run** — full (or resumable) schedule for the chosen configs; generate all figures + quantum-specific reporting.
6. **Analysis / write-up** — overlay comparisons, param/wall-clock table, gradient-variance discussion; decide whether to escalate depth/qubits.

---

*Prepared as a design plan; no model code changed yet. Implementation to follow once the §12 decisions are locked.*
