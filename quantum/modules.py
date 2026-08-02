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


def sandwich(opt, in_features, out_features):
    """The one construction every site so far reduces to.

    Section 3 of the plan makes this inevitable rather than incidental: a circuit
    takes one angle per qubit in and returns one expectation per qubit out, so
    every site is the same classical-quantum-classical sandwich and differs only
    in the widths on either side. Holding it fixed is the point of the
    experiment - if each site had its own circuit, a difference between two
    placements would be confounded by two different circuit designs, and the map
    of *where* a VQC can go would stop being a map.

    So the site functions below carry the reasoning, the shapes and the costs,
    and hand the construction here. Sites D and F will not reduce to this - the
    modulation site's multiply branch ends in a sigmoid gate that PauliZ
    expectations in [-1, 1] cannot feed directly, and the source encoder folds
    patches rather than positions - and they get their own bodies when they land.
    """
    return HybridVQC(in_features, out_features,
                     n_qubits=opt.n_qubits, n_layers=opt.vqc_layers)


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
    return sandwich(opt, in_features, out_features)


@register_site('projection')
def build_projection(opt, in_features, out_features):
    """Site B - the channel-encoder projection, and the headline site.

    Replaces Conv3x3(256 -> 16), whose output *is* the latent that gets power
    normalised, masked and put on the channel. A circuit here is producing the
    transmitted symbols themselves, which is the strongest claim available in
    this pipeline; every other site only shapes or selects what a classical
    layer transmits.

    Returns a plain (B, 256) -> (B, 16) module. The caller wraps it in
    models.networks.PerPosition, which folds the 8x8 grid into the batch so all
    64 positions run as one circuit evaluation and the wrapper is shared with
    the matched control. Two consequences worth stating rather than discovering
    later:

    - The circuit sees batch 128*64 = 8192 rather than 128. That is close to
      free here (76 ms/step against 44 at batch 128) because 8-qubit
      statevector simulation is bound by kernel launches, not arithmetic.
    - What it replaces is a 3x3 convolution, so the swap also gives up the
      receptive field. PerPosition documents this; it applies to both swapped
      arms equally, so quantum against matched stays clean.

    Unlike the policy site, no SNR feature is concatenated. The classical
    Conv3x3 does not receive one - the channel state arrives through the mod1
    and mod2 blocks upstream - so feeding it here would hand this arm an input
    the module it replaces never had, and any difference would be partly
    attributable to that rather than to the circuit.

    Parameters (q=8, L=2): pre 2,056 + angles 32 + post 144 = 2,232, against
    36,880 for the convolution. Being 16x smaller is exactly why the matched
    control at 2,272 is not optional here.
    """
    return sandwich(opt, in_features, out_features)


@register_site('decoder')
def build_decoder(opt, in_features, out_features):
    """Site C - the decoder input, and the receive half of a quantum link.

    Replaces netG.mask_conv, Conv3x3(16 -> 256): the first thing the receiver
    does with what came off the channel. Pair it with 'projection' via
    --quantum_site projection,decoder and both ends of the link are circuits,
    which is the strongest claim this pipeline supports.

    Reading it alongside site B matters more than reading it alone. Site B
    constricts the transmitted latent to rank 8 of 16; this site then expands 16
    to 256 through another 8-measurement circuit, so the decoder's features are
    rank 8 of 256. Those do not compound - composing two rank-8 maps is still
    rank 8 - so B has already discarded exactly the dimensions C would discard,
    and C's marginal cost on top of B should be *smaller* than its cost alone.
    That asymmetry is the interesting measurement here.

    What it does add on top of B is a second receptive-field loss. After AWGN
    the received signal is a rank-8 signal plus full-rank noise, and the
    classical Conv3x3 can project onto the signal subspace and suppress the
    noise in the discarded dimensions. Here that projection has to be learned by
    the pre-linear instead. Whether it is, is what this site tests.

    Parameters (q=8, L=2): pre 136 + angles 32 + post 2,304 = 2,472, against
    36,864 for the convolution. Note the sandwich is the other way round from
    site B - almost all of the budget is on the output side.
    """
    return sandwich(opt, in_features, out_features)
