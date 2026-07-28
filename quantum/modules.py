"""Swappable quantum modules, one per insertion site.

Each site module must be a drop-in for the classical module it replaces: same
input and output shapes, so nothing downstream changes. That constraint is what
keeps the classical-vs-quantum comparison honest, because everything except the
swapped module stays identical.

A site factory is registered by name and receives the parsed options plus the
shapes the classical module had:

    @register_site('policy')
    def _build_policy(opt, in_features, out_features):
        ...

`models.networks.build_swappable` calls it when `--quantum_site` names that
site. The classical control for the same site is `MatchedBottleneck`, which
lives in `models/networks.py` because it is classical.

Costs measured on this hardware are in `quantum/README.md`; run
`python -m quantum.smoke_test --bench` to measure them on your own GPU.
"""
import torch.nn as nn

from quantum import register_site
from quantum.layers import HybridVQC, VariationalCircuit     # noqa: F401


@register_site('policy')
def build_policy_gate(opt, in_features, out_features):
    """Site A - the rate-control policy.

    Replaces the MLP that turns pooled encoder features plus the SNR into one
    logit per rate. Downstream, Gumbel-Softmax samples those logits and a
    straight-through estimator carries gradients back, so this site also
    exercises the longest gradient path of any candidate: loss -> decoder ->
    mask -> Gumbel-Softmax -> circuit.

    Cheapest site to run (one circuit per sample, batch 128) and self-contained,
    which is why it comes first: it proves the plumbing before the sites that
    cost more. Its effect on PSNR should be small, since the policy only picks
    how many groups to send rather than what is in them.

    Note that the fine-tuning stage freezes netP, so a circuit here stops
    training for the last quarter of the schedule unless that is changed
    deliberately.
    """
    return HybridVQC(in_features, out_features,
                     n_qubits=opt.n_qubits, n_layers=opt.vqc_layers)
