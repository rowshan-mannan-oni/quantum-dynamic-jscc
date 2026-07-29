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

**Install from GitHub, not PyPI, and always with `--no-deps`:**

```bash
pip install --no-deps git+https://github.com/mit-han-lab/torchquantum.git
pip install opt_einsum torchpack torchdiffeq tqdm dill pathos
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

No version pins are needed. Two things make this work:

- **GitHub over PyPI.** The last PyPI release (0.1.8, Feb 2024) imports `qiskit.providers.aer`,
  a path qiskit removed in 1.0, so it cannot run against a current qiskit at all. GitHub `main`
  is 0.3.0 and uses `qiskit_aer`, which works with qiskit 2.x.
- **`--no-deps`.** The declared dependency list is stale and pulls in `tensorflow`, `pyscf` and
  `qiskit<1.0.0`. None are used by statevector simulation, and `pyscf` has no Windows wheels, so
  a plain `pip install torchquantum` fails outright there while merely wasting a gigabyte and
  risking a CUDA clash elsewhere.

Verified working on **torch 2.5.1+cu124**, RTX 4070 (sm_89), with torchquantum 0.3.0,
qiskit 2.5.1, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.48.0, setuptools 83, numpy 2.2.6.

> **These installs upgrade numpy to 2.x**, because qiskit requires it. Anything compiled against
> the numpy 1.x ABI then fails at import with `numpy.dtype size changed, may indicate binary
> incompatibility` — scikit-image 0.22 does, which is why `requirements.txt` asks for
> `scikit-image>=0.25`. Install the quantum extras *before* concluding the classical side works.
>
> If an older qiskit is already present, uninstall `qiskit`, `qiskit-terra` and `qiskit-aer`
> first: qiskit >=1.0 refuses to run alongside `qiskit-terra`, and `--upgrade` alone does not
> resolve it.
>
> Both arms share one environment, so after changing quantum dependencies re-check the classical
> side too, not just the smoke test:
> `python test_dyna.py --gpu_ids 0 --lambda_reward 1.5e-3 --SNR 10 --num_test 300`

Kaggle already ships a CUDA build of PyTorch, so install the packages above but never the pinned
`torch` from `requirements.txt`.

> If you are stuck on the PyPI release for some reason, it additionally needs
> `qiskit==0.46.3`, `qiskit-aer==0.13.3` and `setuptools<81` (its qiskit path imports
> `pkg_resources`, removed in setuptools 81). Prefer the GitHub install.

## Verify before building anything on it

```bash
python -m quantum.smoke_test            # circuit runs and gradients reach it
python -m quantum.smoke_test --bench    # time each candidate site
python -m quantum.smoke_test --gpu 1    # on a specific card, matching --gpu_ids
```

A successful import proves very little. What matters is that gradients reach the circuit
parameters, because a silent failure there looks exactly like a model that will not learn.

## Measured cost (RTX 4070 Laptop, torchquantum 0.3.0, 8 qubits unless noted)

| Site | Batch | ms/step | Added min/epoch |
|---|---|---|---|
| policy / modulation | 128 | 44 | +0.3 |
| projection | 8192 | 76 | +0.5 |
| projection (4 layers) | 8192 | 131 | +0.9 |

Classical baseline on the same GPU is ~0.5 min/epoch, so a 400-epoch hybrid run at the projection
site takes roughly 6.5 h against 3.4 h classical. Re-run `--bench` on your own GPU rather than
scaling these numbers.

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
