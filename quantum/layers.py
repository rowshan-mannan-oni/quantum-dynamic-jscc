"""Reusable variational-circuit building blocks.

Nothing here knows about JSCC. These are the primitives that the swappable
site modules in `quantum/modules.py` are assembled from, so every site shares
one circuit implementation and one set of design choices.
"""
import math

import torch
import torch.nn as nn
import torchquantum as tq


class VariationalCircuit(tq.QuantumModule):
    """Angle-encoded hardware-efficient ansatz.

    Takes `n_wires` classical values per sample, encodes them as RY rotations,
    applies `n_layers` blocks of trainable RY/RZ rotations followed by a ring of
    CNOTs, and returns the PauliZ expectation on every wire.

    The whole batch runs as one batched statevector, so cost is set far more by
    the number of gates than by the batch size: at 8 qubits a batch of 8192
    costs about the same as a batch of 128.

    Parameters:
        n_wires (int)   -- qubit count. Cost grows as 2**n_wires; 4-8 is sane.
        n_layers (int)  -- ansatz depth. Start shallow, see the note below.
        init_std (float)-- std of the initial rotation angles.

    Depth and barren plateaus: gradients of a random deep circuit vanish
    exponentially with width and depth, so more layers is not safer. Two layers
    and small initial angles keep the circuit near identity, where gradients
    still carry signal.

    Output is in [-1, 1] (an expectation value), so whatever consumes it should
    expect that scale rather than an unbounded activation.
    """

    def __init__(self, n_wires, n_layers=2, init_std=0.01):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers

        self.encoder = tq.GeneralEncoder(
            [{'input_idx': [i], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])

        self.rys = tq.QuantumModuleList()
        self.rzs = tq.QuantumModuleList()
        for _ in range(n_layers * n_wires):
            self.rys.append(tq.RY(has_params=True, trainable=True))
            self.rzs.append(tq.RZ(has_params=True, trainable=True))

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std=0.01):
        """Start near the identity so early gradients do not vanish."""
        with torch.no_grad():
            for gate in list(self.rys) + list(self.rzs):
                gate.params.normal_(0.0, init_std)

    def forward(self, angles):
        """angles: (B, n_wires) -> expectations: (B, n_wires) in [-1, 1]."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=angles.shape[0],
                                device=angles.device)
        self.encoder(qdev, angles)
        k = 0
        for _ in range(self.n_layers):
            for wire in range(self.n_wires):
                self.rys[k](qdev, wires=wire)
                self.rzs[k](qdev, wires=wire)
                k += 1
            for wire in range(self.n_wires):
                tq.functional.cnot(qdev, wires=[wire, (wire + 1) % self.n_wires])
        return self.measure(qdev)


class HybridVQC(nn.Module):
    """Classical -> quantum -> classical sandwich.

        in_features -> Linear -> bounded angles -> VQC -> measure -> Linear -> out_features

    A circuit cannot absorb or emit a high-dimensional vector: encoding costs one
    angle per qubit and measurement yields one expectation per qubit. The linear
    maps on either side are what let a quantum block stand in for a layer whose
    real width is far larger than any tractable qubit count.

    Angles are squashed with tanh and scaled to +/-pi so the encoding cannot
    saturate or wrap, which otherwise flattens gradients.
    """

    def __init__(self, in_features, out_features, n_qubits=8, n_layers=2,
                 init_std=0.01):
        super().__init__()
        self.pre = nn.Linear(in_features, n_qubits)
        self.circuit = VariationalCircuit(n_qubits, n_layers, init_std)
        self.post = nn.Linear(n_qubits, out_features)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def forward(self, x):
        """x: (B, in_features) -> (B, out_features)."""
        angles = torch.tanh(self.pre(x)) * math.pi
        return self.post(self.circuit(angles))

    def n_quantum_parameters(self):
        return sum(p.numel() for p in self.circuit.parameters())

    def extra_repr(self):
        return f'n_qubits={self.n_qubits}, n_layers={self.n_layers}'
