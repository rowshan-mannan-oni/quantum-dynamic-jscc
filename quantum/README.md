# Quantum extensions

Everything quantum lives in this package. The classical pipeline in `models/` is shared by both
arms, so a classical and a hybrid run differ only in which module one factory returns — that is
what makes the comparison fair rather than merely asserted.

```
quantum/
  layers.py       VariationalCircuit, HybridVQC  — circuit primitives
  modules.py      per-site drop-in modules (to be filled in)
  smoke_test.py   verify the install actually works
  __init__.py     site registry
```

Select a site at run time:

```bash
python train_dyna.py --quantum_site none          # fully classical (default)
python train_dyna.py --quantum_site projection --n_qubits 8 --vqc_layers 2
```

An unknown site name raises rather than falling back, so a typo cannot quietly train a classical
model that then gets written up as a quantum result.

## Installing TorchQuantum

**Do not `pip install torchquantum` on its own.** Its 0.1.8 dependency list is stale and pulls in
`tensorflow`, `pyscf` (no Windows wheels, needs a C++ toolchain) and `qiskit<1.0.0`, none of which
statevector simulation uses. Install the package without dependencies and add only what is
actually imported:

```bash
pip install torchquantum --no-deps
pip install opt_einsum torchpack torchdiffeq tqdm dill pathos
pip install "qiskit==0.46.3" qiskit-ibm-runtime "qiskit-aer==0.13.3"
pip install "setuptools<81"      # torchquantum's qiskit path still imports pkg_resources
```

Pins worth knowing:

| Pin | Why |
|---|---|
| `--no-deps` | avoids `tensorflow` and `pyscf` |
| `qiskit==0.46.3`, `qiskit-aer==0.13.3` | 0.1.8 imports `qiskit.providers.aer`, removed in qiskit 1.x |
| `setuptools<81` | `pkg_resources` was removed in 81 |

Verified working with **torch 2.5.1+cu124** on an RTX 4070 (sm_89). Kaggle already ships a CUDA
build of PyTorch, so install only the packages above there — never the pinned `torch` from
`requirements.txt`.

## Verify before building anything on it

```bash
python -m quantum.smoke_test            # circuit runs and gradients reach it
python -m quantum.smoke_test --bench    # time each candidate site
```

A successful import proves very little. What matters is that gradients reach the circuit
parameters, because a silent failure there looks exactly like a model that will not learn.

## Measured cost (RTX 4070 Laptop, 8 qubits unless noted)

| Site | Batch | ms/step | Added min/epoch |
|---|---|---|---|
| policy / modulation | 128 | 62 | +0.4 |
| projection | 8192 | 86 | +0.6 |
| projection (4 layers) | 8192 | 156 | +1.0 |

Classical baseline on the same GPU is ~0.5 min/epoch, so a 400-epoch hybrid run at the projection
site takes roughly 7 h against 3.4 h classical.

The useful surprise: a batch of 8192 costs about the same as 128. At this scale the simulation is
bound by kernel launches rather than arithmetic, so folding the 64 spatial positions of an
8x8 feature map into the batch dimension is nearly free. One circuit execution per forward
covers every position, which is what makes the channel-encoder site affordable at all.

The first call is much slower (~1.8 s) from CUDA warm-up. Always discard it when timing.

## Design notes

- **Shallow circuits.** Gradients of a random deep circuit vanish exponentially with width and
  depth, so more layers is not a safety margin. `VariationalCircuit` defaults to 2 layers and
  initialises angles near zero, keeping the circuit close to identity where gradients survive.
- **Bounded encoding.** Angles are `tanh(x) * pi`, so the encoding cannot saturate or wrap.
- **Measurements are in [-1, 1].** Whatever consumes a circuit output should expect that scale.
- **A circuit cannot carry a wide vector.** Encoding costs one angle per qubit and measurement
  returns one expectation per qubit, so a quantum block standing in for a wide layer needs
  classical projections on both sides. That is what `HybridVQC` is.

## Fairness

Comparisons need a third arm besides classical and quantum: a **classical-matched** control with
roughly the quantum parameter count, so a difference cannot be explained by "the layer got
smaller". Hold seeds, epochs, data order and SNR range identical across arms, report mean and
spread over several seeds, and log parameter count, wall-clock per epoch and circuit gradient
variance for each. Gradient variance collapsing toward zero is the early warning for a barren
plateau.
