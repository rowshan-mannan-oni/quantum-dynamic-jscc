"""Swappable quantum modules, one per insertion site.

Each site module must be a drop-in for the classical module it replaces: same
input and output shapes, so nothing downstream changes. That constraint is what
keeps the classical-vs-quantum comparison honest, because everything except the
swapped module stays identical.

Register a site like this:

    @register_site('projection')
    def _build_projection(opt, in_dim, out_dim):
        return QuantumProjection(in_dim, out_dim, n_qubits=opt.n_qubits, ...)

No sites are implemented yet; the primitives they will be built from live in
`quantum/layers.py`, and `quantum/README.md` records the cost of each candidate
site as measured on this hardware.
"""
from quantum import register_site                            # noqa: F401
from quantum.layers import HybridVQC, VariationalCircuit     # noqa: F401
